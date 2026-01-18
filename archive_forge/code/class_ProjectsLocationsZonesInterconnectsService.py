from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgenetwork.v1 import edgenetwork_v1_messages as messages
class ProjectsLocationsZonesInterconnectsService(base_api.BaseApiService):
    """Service class for the projects_locations_zones_interconnects resource."""
    _NAME = 'projects_locations_zones_interconnects'

    def __init__(self, client):
        super(EdgenetworkV1.ProjectsLocationsZonesInterconnectsService, self).__init__(client)
        self._upload_configs = {}

    def Diagnose(self, request, global_params=None):
        """Get the diagnostics of a single interconnect resource.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectsDiagnoseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiagnoseInterconnectResponse) The response message.
      """
        config = self.GetMethodConfig('Diagnose')
        return self._RunMethod(config, request, global_params=global_params)
    Diagnose.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnects/{interconnectsId}:diagnose', http_method='GET', method_id='edgenetwork.projects.locations.zones.interconnects.diagnose', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:diagnose', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectsDiagnoseRequest', response_type_name='DiagnoseInterconnectResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Interconnect.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Interconnect) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnects/{interconnectsId}', http_method='GET', method_id='edgenetwork.projects.locations.zones.interconnects.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectsGetRequest', response_type_name='Interconnect', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Interconnects in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInterconnectsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInterconnectsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/interconnects', http_method='GET', method_id='edgenetwork.projects.locations.zones.interconnects.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/interconnects', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesInterconnectsListRequest', response_type_name='ListInterconnectsResponse', supports_download=False)