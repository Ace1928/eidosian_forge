from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class ProjectsInstancesClustersService(base_api.BaseApiService):
    """Service class for the projects_instances_clusters resource."""
    _NAME = 'projects_instances_clusters'

    def __init__(self, client):
        super(BigtableadminV2.ProjectsInstancesClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a cluster within an instance. Note that exactly one of Cluster.serve_nodes and Cluster.cluster_config.cluster_autoscaling_config can be set. If serve_nodes is set to non-zero, then the cluster is manually scaled. If cluster_config.cluster_autoscaling_config is non-empty, then autoscaling is enabled.

      Args:
        request: (BigtableadminProjectsInstancesClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters', http_method='POST', method_id='bigtableadmin.projects.instances.clusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['clusterId'], relative_path='v2/{+parent}/clusters', request_field='cluster', request_type_name='BigtableadminProjectsInstancesClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a cluster from an instance.

      Args:
        request: (BigtableadminProjectsInstancesClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}', http_method='DELETE', method_id='bigtableadmin.projects.instances.clusters.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesClustersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a cluster.

      Args:
        request: (BigtableadminProjectsInstancesClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}', http_method='GET', method_id='bigtableadmin.projects.instances.clusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesClustersGetRequest', response_type_name='Cluster', supports_download=False)

    def List(self, request, global_params=None):
        """Lists information about clusters in an instance.

      Args:
        request: (BigtableadminProjectsInstancesClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters', http_method='GET', method_id='bigtableadmin.projects.instances.clusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageToken'], relative_path='v2/{+parent}/clusters', request_field='', request_type_name='BigtableadminProjectsInstancesClustersListRequest', response_type_name='ListClustersResponse', supports_download=False)

    def PartialUpdateCluster(self, request, global_params=None):
        """Partially updates a cluster within a project. This method is the preferred way to update a Cluster. To enable and update autoscaling, set cluster_config.cluster_autoscaling_config. When autoscaling is enabled, serve_nodes is treated as an OUTPUT_ONLY field, meaning that updates to it are ignored. Note that an update cannot simultaneously set serve_nodes to non-zero and cluster_config.cluster_autoscaling_config to non-empty, and also specify both in the update_mask. To disable autoscaling, clear cluster_config.cluster_autoscaling_config, and explicitly set a serve_node count via the update_mask.

      Args:
        request: (BigtableadminProjectsInstancesClustersPartialUpdateClusterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PartialUpdateCluster')
        return self._RunMethod(config, request, global_params=global_params)
    PartialUpdateCluster.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}', http_method='PATCH', method_id='bigtableadmin.projects.instances.clusters.partialUpdateCluster', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='cluster', request_type_name='BigtableadminProjectsInstancesClustersPartialUpdateClusterRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a cluster within an instance. Note that UpdateCluster does not support updating cluster_config.cluster_autoscaling_config. In order to update it, you must use PartialUpdateCluster.

      Args:
        request: (Cluster) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}', http_method='PUT', method_id='bigtableadmin.projects.instances.clusters.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='<request>', request_type_name='Cluster', response_type_name='Operation', supports_download=False)