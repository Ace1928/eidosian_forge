from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
class ProjectsLocationsGlobalManagedZonesService(base_api.BaseApiService):
    """Service class for the projects_locations_global_managedZones resource."""
    _NAME = 'projects_locations_global_managedZones'

    def __init__(self, client):
        super(ConnectorsV1.ProjectsLocationsGlobalManagedZonesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ManagedZone in a given project and location.

      Args:
        request: (ConnectorsProjectsLocationsGlobalManagedZonesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/managedZones', http_method='POST', method_id='connectors.projects.locations.global.managedZones.create', ordered_params=['parent'], path_params=['parent'], query_params=['managedZoneId'], relative_path='v1/{+parent}/managedZones', request_field='managedZone', request_type_name='ConnectorsProjectsLocationsGlobalManagedZonesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ManagedZone.

      Args:
        request: (ConnectorsProjectsLocationsGlobalManagedZonesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/managedZones/{managedZonesId}', http_method='DELETE', method_id='connectors.projects.locations.global.managedZones.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ConnectorsProjectsLocationsGlobalManagedZonesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ManagedZone.

      Args:
        request: (ConnectorsProjectsLocationsGlobalManagedZonesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/managedZones/{managedZonesId}', http_method='GET', method_id='connectors.projects.locations.global.managedZones.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ConnectorsProjectsLocationsGlobalManagedZonesGetRequest', response_type_name='ManagedZone', supports_download=False)

    def List(self, request, global_params=None):
        """List ManagedZones in a given project.

      Args:
        request: (ConnectorsProjectsLocationsGlobalManagedZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListManagedZonesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/managedZones', http_method='GET', method_id='connectors.projects.locations.global.managedZones.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/managedZones', request_field='', request_type_name='ConnectorsProjectsLocationsGlobalManagedZonesListRequest', response_type_name='ListManagedZonesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ManagedZone.

      Args:
        request: (ConnectorsProjectsLocationsGlobalManagedZonesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/managedZones/{managedZonesId}', http_method='PATCH', method_id='connectors.projects.locations.global.managedZones.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='managedZone', request_type_name='ConnectorsProjectsLocationsGlobalManagedZonesPatchRequest', response_type_name='Operation', supports_download=False)