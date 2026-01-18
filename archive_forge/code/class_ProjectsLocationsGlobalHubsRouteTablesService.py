from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
class ProjectsLocationsGlobalHubsRouteTablesService(base_api.BaseApiService):
    """Service class for the projects_locations_global_hubs_routeTables resource."""
    _NAME = 'projects_locations_global_hubs_routeTables'

    def __init__(self, client):
        super(NetworkconnectivityV1beta.ProjectsLocationsGlobalHubsRouteTablesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details about a Network Connectivity Center route table.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaRouteTable) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}/routeTables/{routeTablesId}', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.routeTables.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesGetRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaRouteTable', supports_download=False)

    def List(self, request, global_params=None):
        """Lists route tables in a given hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListRouteTablesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}/routeTables', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.routeTables.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/routeTables', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesListRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListRouteTablesResponse', supports_download=False)