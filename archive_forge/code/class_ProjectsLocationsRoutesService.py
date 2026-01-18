from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsLocationsRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_routes resource."""
    _NAME = 'projects_locations_routes'

    def __init__(self, client):
        super(RunV1.ProjectsLocationsRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get information about a route.

      Args:
        request: (RunProjectsLocationsRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Route) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/routes/{routesId}', http_method='GET', method_id='run.projects.locations.routes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsRoutesGetRequest', response_type_name='Route', supports_download=False)

    def List(self, request, global_params=None):
        """List routes.

      Args:
        request: (RunProjectsLocationsRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/routes', http_method='GET', method_id='run.projects.locations.routes.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='v1/{+parent}/routes', request_field='', request_type_name='RunProjectsLocationsRoutesListRequest', response_type_name='ListRoutesResponse', supports_download=False)