from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgenetwork.v1alpha1 import edgenetwork_v1alpha1_messages as messages
class ProjectsLocationsZonesRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_zones_routes resource."""
    _NAME = 'projects_locations_zones_routes'

    def __init__(self, client):
        super(EdgenetworkV1alpha1.ProjectsLocationsZonesRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Route, in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routes', http_method='POST', method_id='edgenetwork.projects.locations.zones.routes.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'routeId'], relative_path='v1alpha1/{+parent}/routes', request_field='route', request_type_name='EdgenetworkProjectsLocationsZonesRoutesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Route.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routes/{routesId}', http_method='DELETE', method_id='edgenetwork.projects.locations.zones.routes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Route.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Route) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routes/{routesId}', http_method='GET', method_id='edgenetwork.projects.locations.zones.routes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutesGetRequest', response_type_name='Route', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Routes in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routes', http_method='GET', method_id='edgenetwork.projects.locations.zones.routes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/routes', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutesListRequest', response_type_name='ListRoutesResponse', supports_download=False)