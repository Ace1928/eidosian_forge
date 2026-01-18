from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsClustersNodesService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_clusters_nodes resource."""
    _NAME = 'projects_locations_privateClouds_clusters_nodes'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsClustersNodesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single node.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsClustersNodesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Node) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/clusters/{clustersId}/nodes/{nodesId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.clusters.nodes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsClustersNodesGetRequest', response_type_name='Node', supports_download=False)

    def List(self, request, global_params=None):
        """Lists nodes in a given cluster.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsClustersNodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNodesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/clusters/{clustersId}/nodes', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.clusters.nodes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/nodes', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsClustersNodesListRequest', response_type_name='ListNodesResponse', supports_download=False)