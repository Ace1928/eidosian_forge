from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class DeploymentsDevicesService(base_api.BaseApiService):
    """Service class for the deployments_devices resource."""
    _NAME = 'deployments_devices'

    def __init__(self, client):
        super(SasportalV1alpha1.DeploymentsDevicesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a device.

      Args:
        request: (SasportalDeploymentsDevicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/deployments/{deploymentsId}/devices/{devicesId}', http_method='DELETE', method_id='sasportal.deployments.devices.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalDeploymentsDevicesDeleteRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a device.

      Args:
        request: (SasportalDeploymentsDevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/deployments/{deploymentsId}/devices/{devicesId}', http_method='GET', method_id='sasportal.deployments.devices.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalDeploymentsDevicesGetRequest', response_type_name='SasPortalDevice', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves a device under another node or customer.

      Args:
        request: (SasportalDeploymentsDevicesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/deployments/{deploymentsId}/devices/{devicesId}:move', http_method='POST', method_id='sasportal.deployments.devices.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:move', request_field='sasPortalMoveDeviceRequest', request_type_name='SasportalDeploymentsDevicesMoveRequest', response_type_name='SasPortalOperation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a device.

      Args:
        request: (SasportalDeploymentsDevicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/deployments/{deploymentsId}/devices/{devicesId}', http_method='PATCH', method_id='sasportal.deployments.devices.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='sasPortalDevice', request_type_name='SasportalDeploymentsDevicesPatchRequest', response_type_name='SasPortalDevice', supports_download=False)

    def SignDevice(self, request, global_params=None):
        """Signs a device.

      Args:
        request: (SasportalDeploymentsDevicesSignDeviceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('SignDevice')
        return self._RunMethod(config, request, global_params=global_params)
    SignDevice.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/deployments/{deploymentsId}/devices/{devicesId}:signDevice', http_method='POST', method_id='sasportal.deployments.devices.signDevice', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:signDevice', request_field='sasPortalSignDeviceRequest', request_type_name='SasportalDeploymentsDevicesSignDeviceRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def UpdateSigned(self, request, global_params=None):
        """Updates a signed device.

      Args:
        request: (SasportalDeploymentsDevicesUpdateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('UpdateSigned')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSigned.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/deployments/{deploymentsId}/devices/{devicesId}:updateSigned', http_method='PATCH', method_id='sasportal.deployments.devices.updateSigned', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:updateSigned', request_field='sasPortalUpdateSignedDeviceRequest', request_type_name='SasportalDeploymentsDevicesUpdateSignedRequest', response_type_name='SasPortalDevice', supports_download=False)