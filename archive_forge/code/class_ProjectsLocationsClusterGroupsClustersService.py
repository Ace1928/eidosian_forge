from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
class ProjectsLocationsClusterGroupsClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_clusterGroups_clusters resource."""
    _NAME = 'projects_locations_clusterGroups_clusters'

    def __init__(self, client):
        super(SddcV1alpha1.ProjectsLocationsClusterGroupsClustersService, self).__init__(client)
        self._upload_configs = {}

    def AddNodes(self, request, global_params=None):
        """Add bare metal nodes to a cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersAddNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('AddNodes')
        return self._RunMethod(config, request, global_params=global_params)
    AddNodes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters/{clustersId}:addNodes', http_method='POST', method_id='sddc.projects.locations.clusterGroups.clusters.addNodes', ordered_params=['cluster'], path_params=['cluster'], query_params=[], relative_path='v1alpha1/{+cluster}:addNodes', request_field='addNodesRequest', request_type_name='SddcProjectsLocationsClusterGroupsClustersAddNodesRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new cluster in a given cluster group. The creation is asynchronous. You can check the returned operation to track its progress. When the operation successfully completes, the cluster has a a **READY** status and is fully functional. The returned operation is automatically deleted after a few hours, so there is no need to call `operations.delete`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters', http_method='POST', method_id='sddc.projects.locations.clusterGroups.clusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['clusterId', 'managementCluster'], relative_path='v1alpha1/{+parent}/clusters', request_field='cluster', request_type_name='SddcProjectsLocationsClusterGroupsClustersCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters/{clustersId}', http_method='DELETE', method_id='sddc.projects.locations.clusterGroups.clusters.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsClustersDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters/{clustersId}', http_method='GET', method_id='sddc.projects.locations.clusterGroups.clusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsClustersGetRequest', response_type_name='Cluster', supports_download=False)

    def List(self, request, global_params=None):
        """Lists clusters in a given cluster group.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters', http_method='GET', method_id='sddc.projects.locations.clusterGroups.clusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/clusters', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsClustersListRequest', response_type_name='ListClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates labels of a specific cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters/{clustersId}', http_method='PATCH', method_id='sddc.projects.locations.clusterGroups.clusters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='cluster', request_type_name='SddcProjectsLocationsClusterGroupsClustersPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def RemoveNodes(self, request, global_params=None):
        """Remove bare metal nodes from a cluster.

      Args:
        request: (SddcProjectsLocationsClusterGroupsClustersRemoveNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('RemoveNodes')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveNodes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/clusters/{clustersId}:removeNodes', http_method='POST', method_id='sddc.projects.locations.clusterGroups.clusters.removeNodes', ordered_params=['cluster'], path_params=['cluster'], query_params=[], relative_path='v1alpha1/{+cluster}:removeNodes', request_field='removeNodesRequest', request_type_name='SddcProjectsLocationsClusterGroupsClustersRemoveNodesRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)