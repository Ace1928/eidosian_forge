from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
class ProjectsLocationsRegistriesGroupsDevicesService(base_api.BaseApiService):
    """Service class for the projects_locations_registries_groups_devices resource."""
    _NAME = 'projects_locations_registries_groups_devices'

    def __init__(self, client):
        super(CloudiotV1.ProjectsLocationsRegistriesGroupsDevicesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List devices in a device registry.

      Args:
        request: (CloudiotProjectsLocationsRegistriesGroupsDevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDevicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/groups/{groupsId}/devices', http_method='GET', method_id='cloudiot.projects.locations.registries.groups.devices.list', ordered_params=['parent'], path_params=['parent'], query_params=['deviceIds', 'deviceNumIds', 'fieldMask', 'gatewayListOptions_associationsDeviceId', 'gatewayListOptions_associationsGatewayId', 'gatewayListOptions_gatewayType', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/devices', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesGroupsDevicesListRequest', response_type_name='ListDevicesResponse', supports_download=False)