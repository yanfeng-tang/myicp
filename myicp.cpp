#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <pcl/search/impl/kdtree.hpp>
#include <vector>

using namespace std;

void ViusalCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, string name) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer);
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(
      cloud, 0, 0, 255);
  viewer->addPointCloud(cloud, cloud_color, name);

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

void downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr CloudFilter,
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float l) {
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(l, l, l);
  sor.filter(*CloudFilter);
}

void ComputeAvePoint(pcl::PointXYZ &p,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud) {
  float x = 0.0;
  float y = 0.0;
  float z = 0.0;
  int n = Cloud->size();
  if (n < 100) {
    cout << "not enough points" << endl;
    return;
  }
  for (size_t i = 0; i < Cloud->size(); i++) {
    x += (*Cloud)[i].x;
    y += (*Cloud)[i].y;
    z += (*Cloud)[i].z;
  }
  x = x / n;
  y = y / n;
  z = z / n;
  p.x = x;
  p.y = y;
  p.z = z;
}

void demeanpoint(pcl::PointXYZ &p, pcl::PointXYZ &meanp, pcl::PointXYZ &pt) {
  p.x = pt.x - meanp.x;
  p.y = pt.y - meanp.y;
  p.z = pt.z - meanp.z;
}

int main(int argc, char **argv) {
  cout << "This is my icp" << endl;

  // pcd -> pcl point
  string file1 = "/home/tangyf/myicp/src/PCDdata/first.pcd";
  string file2 = "/home/tangyf/myicp/src/PCDdata/second.pcd";
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_first(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(file1, *cloud_first);
  pcl::io::loadPCDFile(file2, *cloud_target);

  // downsample
  pcl::PointCloud<pcl::PointXYZ>::Ptr CloudFirstFilter(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr CloudTargetFilter(
      new pcl::PointCloud<pcl::PointXYZ>);

  downsample(CloudFirstFilter, cloud_first, 0.1f);
  // downsample(CloudTargetFilter, cloud_target, 0.1f);

  // cout << "cloud size is:" << cloud_first->size() << endl;
  // cout << "cloudafterfilt size is:" << CloudFirstFilter->size() << endl;
  // ViusalCloud(CloudFirstFilter, "CloudAfterFilt");
  // ViusalCloud(cloud_first, "CloudBeforeFilt");

  // compute average points ps and pt
  // pcl::PointXYZ Ps, Pt;
  // pcl::PointXYZ tempPs, tempPt;
  // ComputeAvePoint(Ps, CloudFirstFilter);
  // ComputeAvePoint(Pt, CloudTargetFilter);

  // cout << "Ps:" << Ps.x << Ps.y << Ps.z << endl;

  // construct kd_tree
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud_target);

  int maxiter = 10000;
  float minloss = 0.1;
  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = Eigen::Vector3f::Zero();
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_final = Eigen::Matrix4f::Identity();
  Eigen::Vector3f p, pt;
  int iter = 0;
  float loss = INT_MAX;
  while (iter < maxiter && loss > minloss) {
    iter++;
    // search nearest k points
    vector<int> NearPoints;
    pcl::transformPointCloud(*CloudFirstFilter, *CloudFirstFilter, T);
    Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
    float temploss = 0.0;
    for (size_t i = 0; i < CloudFirstFilter->size(); i++) {
      vector<int> IndexOfPoints(1);
      vector<float> ptKNNSqDis(1);
      kdtree.nearestKSearch((*CloudFirstFilter)[i], 1, IndexOfPoints,
                            ptKNNSqDis);
      int index = IndexOfPoints[0];
      temploss += sqrt(ptKNNSqDis[0]);
      NearPoints.push_back(index);
      // compute H
      // demeanpoint(tempPs, Ps, (*CloudFirstFilter)[i]);
      // demeanpoint(tempPt, Pt, (*CloudTargetFilter)[index]);

      // p << tempPs.x, tempPs.y, tempPs.z;
      // pt << tempPt.x, tempPt.y, tempPt.z;

      // W += p * pt.transpose();
    }
    temploss /= CloudFirstFilter->size();
    if (loss - temploss < 0.0001) break;
    pcl::copyPointCloud(*cloud_target, NearPoints, *CloudTargetFilter);

    // compute average points
    Eigen::Vector4f centroidPs, centroidPt;
    pcl::compute3DCentroid(*CloudFirstFilter, centroidPs);
    pcl::compute3DCentroid(*CloudTargetFilter, centroidPt);

    pcl::PointCloud<pcl::PointXYZ>::Ptr demeanCloudFirst(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr demeanCloudTarget(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::demeanPointCloud(*CloudFirstFilter, centroidPs, *demeanCloudFirst);
    pcl::demeanPointCloud(*CloudTargetFilter, centroidPt, *demeanCloudTarget);

    // Eigen::MatrixXf CloudFirstdemean, CloudTargetdemean;
    // pcl::demeanPointCloud(*CloudFirstFilter, centroidPs, CloudFirstdemean);
    // pcl::demeanPointCloud(*CloudTargetFilter, centroidPt, CloudTargetdemean);
    // construct W
    for (size_t i = 0; i < demeanCloudFirst->size(); i++) {
      p << (*demeanCloudFirst)[i].x, (*demeanCloudFirst)[i].y,
          (*demeanCloudFirst)[i].z;
      pt << (*demeanCloudTarget)[i].x, (*demeanCloudTarget)[i].y,
          (*demeanCloudTarget)[i].z;
      W += p * pt.transpose();
    }
    // Eigen::Matrix3f W1 =
    //     (CloudFirstdemean * CloudTargetdemean.transpose()).topLeftCorner(3,
    //     3);
    // cout << "W" << W << endl;
    // cout << "W1" << W1 << endl;

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
        W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    R = V * U.transpose();
    t = centroidPt.head(3) - R * centroidPs.head(3);
    T << R, t, 0, 0, 0, 1;
    T_final = T * T_final;

    loss = temploss;
    cout << "iter" << iter << " loss is " << loss << endl;
  }

  // icp

  // visual
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer("cloud_view"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::transformPointCloud(*cloud_first, *cloud_first, T_final);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(
      cloud_first, 0, 0, 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color1(
      cloud_target, 255, 0, 0);
  viewer->addPointCloud(cloud_first, cloud_color, "visual_cloud");
  viewer->addPointCloud(cloud_target, cloud_color1, "");

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

  return 0;
}