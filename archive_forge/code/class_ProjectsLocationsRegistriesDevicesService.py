from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
class ProjectsLocationsRegistriesDevicesService(base_api.BaseApiService):
    """Service class for the projects_locations_registries_devices resource."""
    _NAME = 'projects_locations_registries_devices'

    def __init__(self, client):
        super(CloudiotV1.ProjectsLocationsRegistriesDevicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a device in a device registry.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Device) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices', http_method='POST', method_id='cloudiot.projects.locations.registries.devices.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/devices', request_field='device', request_type_name='CloudiotProjectsLocationsRegistriesDevicesCreateRequest', response_type_name='Device', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a device.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices/{devicesId}', http_method='DELETE', method_id='cloudiot.projects.locations.registries.devices.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesDevicesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a device.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Device) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices/{devicesId}', http_method='GET', method_id='cloudiot.projects.locations.registries.devices.get', ordered_params=['name'], path_params=['name'], query_params=['fieldMask'], relative_path='v1/{+name}', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesDevicesGetRequest', response_type_name='Device', supports_download=False)

    def List(self, request, global_params=None):
        """List devices in a device registry.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDevicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices', http_method='GET', method_id='cloudiot.projects.locations.registries.devices.list', ordered_params=['parent'], path_params=['parent'], query_params=['deviceIds', 'deviceNumIds', 'fieldMask', 'gatewayListOptions_associationsDeviceId', 'gatewayListOptions_associationsGatewayId', 'gatewayListOptions_gatewayType', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/devices', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesDevicesListRequest', response_type_name='ListDevicesResponse', supports_download=False)

    def ModifyCloudToDeviceConfig(self, request, global_params=None):
        """Modifies the configuration for the device, which is eventually sent from the Cloud IoT Core servers. Returns the modified configuration version and its metadata.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesModifyCloudToDeviceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceConfig) The response message.
      """
        config = self.GetMethodConfig('ModifyCloudToDeviceConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyCloudToDeviceConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices/{devicesId}:modifyCloudToDeviceConfig', http_method='POST', method_id='cloudiot.projects.locations.registries.devices.modifyCloudToDeviceConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:modifyCloudToDeviceConfig', request_field='modifyCloudToDeviceConfigRequest', request_type_name='CloudiotProjectsLocationsRegistriesDevicesModifyCloudToDeviceConfigRequest', response_type_name='DeviceConfig', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a device.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Device) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices/{devicesId}', http_method='PATCH', method_id='cloudiot.projects.locations.registries.devices.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='device', request_type_name='CloudiotProjectsLocationsRegistriesDevicesPatchRequest', response_type_name='Device', supports_download=False)

    def SendCommandToDevice(self, request, global_params=None):
        """Sends a command to the specified device. In order for a device to be able to receive commands, it must: 1) be connected to Cloud IoT Core using the MQTT protocol, and 2) be subscribed to the group of MQTT topics specified by /devices/{device-id}/commands/#. This subscription will receive commands at the top-level topic /devices/{device-id}/commands as well as commands for subfolders, like /devices/{device-id}/commands/subfolder. Note that subscribing to specific subfolders is not supported. If the command could not be delivered to the device, this method will return an error; in particular, if the device is not subscribed, this method will return FAILED_PRECONDITION. Otherwise, this method will return OK. If the subscription is QoS 1, at least once delivery will be guaranteed; for QoS 0, no acknowledgment will be expected from the device.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesSendCommandToDeviceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SendCommandToDeviceResponse) The response message.
      """
        config = self.GetMethodConfig('SendCommandToDevice')
        return self._RunMethod(config, request, global_params=global_params)
    SendCommandToDevice.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices/{devicesId}:sendCommandToDevice', http_method='POST', method_id='cloudiot.projects.locations.registries.devices.sendCommandToDevice', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:sendCommandToDevice', request_field='sendCommandToDeviceRequest', request_type_name='CloudiotProjectsLocationsRegistriesDevicesSendCommandToDeviceRequest', response_type_name='SendCommandToDeviceResponse', supports_download=False)