from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
class ProjectsLocationsGlobalHubsRouteTablesRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_global_hubs_routeTables_routes resource."""
    _NAME = 'projects_locations_global_hubs_routeTables_routes'

    def __init__(self, client):
        super(NetworkconnectivityV1beta.ProjectsLocationsGlobalHubsRouteTablesRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details about the specified route.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaRoute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}/routeTables/{routeTablesId}/routes/{routesId}', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.routeTables.routes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesRoutesGetRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaRoute', supports_download=False)

    def List(self, request, global_params=None):
        """Lists routes in a given route table.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}/routeTables/{routeTablesId}/routes', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.routeTables.routes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/routes', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsRouteTablesRoutesListRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListRoutesResponse', supports_download=False)