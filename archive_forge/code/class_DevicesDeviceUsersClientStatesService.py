from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class DevicesDeviceUsersClientStatesService(base_api.BaseApiService):
    """Service class for the devices_deviceUsers_clientStates resource."""
    _NAME = 'devices_deviceUsers_clientStates'

    def __init__(self, client):
        super(CloudidentityV1.DevicesDeviceUsersClientStatesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the client state for the device user.

      Args:
        request: (CloudidentityDevicesDeviceUsersClientStatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1ClientState) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}/clientStates/{clientStatesId}', http_method='GET', method_id='cloudidentity.devices.deviceUsers.clientStates.get', ordered_params=['name'], path_params=['name'], query_params=['customer'], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityDevicesDeviceUsersClientStatesGetRequest', response_type_name='GoogleAppsCloudidentityDevicesV1ClientState', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the client states for the given search query.

      Args:
        request: (CloudidentityDevicesDeviceUsersClientStatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1ListClientStatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}/clientStates', http_method='GET', method_id='cloudidentity.devices.deviceUsers.clientStates.list', ordered_params=['parent'], path_params=['parent'], query_params=['customer', 'filter', 'orderBy', 'pageToken'], relative_path='v1/{+parent}/clientStates', request_field='', request_type_name='CloudidentityDevicesDeviceUsersClientStatesListRequest', response_type_name='GoogleAppsCloudidentityDevicesV1ListClientStatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the client state for the device user **Note**: This method is available only to customers who have one of the following SKUs: Enterprise Standard, Enterprise Plus, Enterprise for Education, and Cloud Identity Premium.

      Args:
        request: (CloudidentityDevicesDeviceUsersClientStatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}/clientStates/{clientStatesId}', http_method='PATCH', method_id='cloudidentity.devices.deviceUsers.clientStates.patch', ordered_params=['name'], path_params=['name'], query_params=['customer', 'updateMask'], relative_path='v1/{+name}', request_field='googleAppsCloudidentityDevicesV1ClientState', request_type_name='CloudidentityDevicesDeviceUsersClientStatesPatchRequest', response_type_name='Operation', supports_download=False)