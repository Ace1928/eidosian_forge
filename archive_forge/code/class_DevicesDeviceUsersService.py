from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class DevicesDeviceUsersService(base_api.BaseApiService):
    """Service class for the devices_deviceUsers resource."""
    _NAME = 'devices_deviceUsers'

    def __init__(self, client):
        super(CloudidentityV1.DevicesDeviceUsersService, self).__init__(client)
        self._upload_configs = {}

    def Approve(self, request, global_params=None):
        """Approves device to access user data.

      Args:
        request: (CloudidentityDevicesDeviceUsersApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Approve')
        return self._RunMethod(config, request, global_params=global_params)
    Approve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}:approve', http_method='POST', method_id='cloudidentity.devices.deviceUsers.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='googleAppsCloudidentityDevicesV1ApproveDeviceUserRequest', request_type_name='CloudidentityDevicesDeviceUsersApproveRequest', response_type_name='Operation', supports_download=False)

    def Block(self, request, global_params=None):
        """Blocks device from accessing user data.

      Args:
        request: (CloudidentityDevicesDeviceUsersBlockRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Block')
        return self._RunMethod(config, request, global_params=global_params)
    Block.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}:block', http_method='POST', method_id='cloudidentity.devices.deviceUsers.block', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:block', request_field='googleAppsCloudidentityDevicesV1BlockDeviceUserRequest', request_type_name='CloudidentityDevicesDeviceUsersBlockRequest', response_type_name='Operation', supports_download=False)

    def CancelWipe(self, request, global_params=None):
        """Cancels an unfinished user account wipe. This operation can be used to cancel device wipe in the gap between the wipe operation returning success and the device being wiped.

      Args:
        request: (CloudidentityDevicesDeviceUsersCancelWipeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CancelWipe')
        return self._RunMethod(config, request, global_params=global_params)
    CancelWipe.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}:cancelWipe', http_method='POST', method_id='cloudidentity.devices.deviceUsers.cancelWipe', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancelWipe', request_field='googleAppsCloudidentityDevicesV1CancelWipeDeviceUserRequest', request_type_name='CloudidentityDevicesDeviceUsersCancelWipeRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified DeviceUser. This also revokes the user's access to device data.

      Args:
        request: (CloudidentityDevicesDeviceUsersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}', http_method='DELETE', method_id='cloudidentity.devices.deviceUsers.delete', ordered_params=['name'], path_params=['name'], query_params=['customer'], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityDevicesDeviceUsersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified DeviceUser.

      Args:
        request: (CloudidentityDevicesDeviceUsersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1DeviceUser) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}', http_method='GET', method_id='cloudidentity.devices.deviceUsers.get', ordered_params=['name'], path_params=['name'], query_params=['customer'], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityDevicesDeviceUsersGetRequest', response_type_name='GoogleAppsCloudidentityDevicesV1DeviceUser', supports_download=False)

    def List(self, request, global_params=None):
        """Lists/Searches DeviceUsers.

      Args:
        request: (CloudidentityDevicesDeviceUsersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1ListDeviceUsersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers', http_method='GET', method_id='cloudidentity.devices.deviceUsers.list', ordered_params=['parent'], path_params=['parent'], query_params=['customer', 'filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/deviceUsers', request_field='', request_type_name='CloudidentityDevicesDeviceUsersListRequest', response_type_name='GoogleAppsCloudidentityDevicesV1ListDeviceUsersResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Looks up resource names of the DeviceUsers associated with the caller's credentials, as well as the properties provided in the request. This method must be called with end-user credentials with the scope: https://www.googleapis.com/auth/cloud-identity.devices.lookup If multiple properties are provided, only DeviceUsers having all of these properties are considered as matches - i.e. the query behaves like an AND. Different platforms require different amounts of information from the caller to ensure that the DeviceUser is uniquely identified. - iOS: No properties need to be passed, the caller's credentials are sufficient to identify the corresponding DeviceUser. - Android: Specifying the 'android_id' field is required. - Desktop: Specifying the 'raw_resource_id' field is required.

      Args:
        request: (CloudidentityDevicesDeviceUsersLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1LookupSelfDeviceUsersResponse) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers:lookup', http_method='GET', method_id='cloudidentity.devices.deviceUsers.lookup', ordered_params=['parent'], path_params=['parent'], query_params=['androidId', 'pageSize', 'pageToken', 'rawResourceId', 'userId'], relative_path='v1/{+parent}:lookup', request_field='', request_type_name='CloudidentityDevicesDeviceUsersLookupRequest', response_type_name='GoogleAppsCloudidentityDevicesV1LookupSelfDeviceUsersResponse', supports_download=False)

    def Wipe(self, request, global_params=None):
        """Wipes the user's account on a device. Other data on the device that is not associated with the user's work account is not affected. For example, if a Gmail app is installed on a device that is used for personal and work purposes, and the user is logged in to the Gmail app with their personal account as well as their work account, wiping the "deviceUser" by their work administrator will not affect their personal account within Gmail or other apps such as Photos.

      Args:
        request: (CloudidentityDevicesDeviceUsersWipeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Wipe')
        return self._RunMethod(config, request, global_params=global_params)
    Wipe.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}/deviceUsers/{deviceUsersId}:wipe', http_method='POST', method_id='cloudidentity.devices.deviceUsers.wipe', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:wipe', request_field='googleAppsCloudidentityDevicesV1WipeDeviceUserRequest', request_type_name='CloudidentityDevicesDeviceUsersWipeRequest', response_type_name='Operation', supports_download=False)