from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class ChromeosdevicesService(base_api.BaseApiService):
    """Service class for the chromeosdevices resource."""
    _NAME = u'chromeosdevices'

    def __init__(self, client):
        super(AdminDirectoryV1.ChromeosdevicesService, self).__init__(client)
        self._upload_configs = {}

    def Action(self, request, global_params=None):
        """Take action on Chrome OS Device.

      Args:
        request: (DirectoryChromeosdevicesActionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryChromeosdevicesActionResponse) The response message.
      """
        config = self.GetMethodConfig('Action')
        return self._RunMethod(config, request, global_params=global_params)
    Action.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.chromeosdevices.action', ordered_params=[u'customerId', u'resourceId'], path_params=[u'customerId', u'resourceId'], query_params=[], relative_path=u'customer/{customerId}/devices/chromeos/{resourceId}/action', request_field=u'chromeOsDeviceAction', request_type_name=u'DirectoryChromeosdevicesActionRequest', response_type_name=u'DirectoryChromeosdevicesActionResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve Chrome OS Device.

      Args:
        request: (DirectoryChromeosdevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (ChromeOsDevice) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.chromeosdevices.get', ordered_params=[u'customerId', u'deviceId'], path_params=[u'customerId', u'deviceId'], query_params=[u'projection'], relative_path=u'customer/{customerId}/devices/chromeos/{deviceId}', request_field='', request_type_name=u'DirectoryChromeosdevicesGetRequest', response_type_name=u'ChromeOsDevice', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve all Chrome OS Devices of a customer (paginated).

      Args:
        request: (DirectoryChromeosdevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (ChromeOsDevices) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.chromeosdevices.list', ordered_params=[u'customerId'], path_params=[u'customerId'], query_params=[u'maxResults', u'orderBy', u'orgUnitPath', u'pageToken', u'projection', u'query', u'sortOrder'], relative_path=u'customer/{customerId}/devices/chromeos', request_field='', request_type_name=u'DirectoryChromeosdevicesListRequest', response_type_name=u'ChromeOsDevices', supports_download=False)

    def MoveDevicesToOu(self, request, global_params=None):
        """Move or insert multiple Chrome OS Devices to organizational unit.

      Args:
        request: (DirectoryChromeosdevicesMoveDevicesToOuRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryChromeosdevicesMoveDevicesToOuResponse) The response message.
      """
        config = self.GetMethodConfig('MoveDevicesToOu')
        return self._RunMethod(config, request, global_params=global_params)
    MoveDevicesToOu.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.chromeosdevices.moveDevicesToOu', ordered_params=[u'customerId', u'orgUnitPath'], path_params=[u'customerId'], query_params=[u'orgUnitPath'], relative_path=u'customer/{customerId}/devices/chromeos/moveDevicesToOu', request_field=u'chromeOsMoveDevicesToOu', request_type_name=u'DirectoryChromeosdevicesMoveDevicesToOuRequest', response_type_name=u'DirectoryChromeosdevicesMoveDevicesToOuResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update Chrome OS Device.

      This method supports patch semantics.

      Args:
        request: (DirectoryChromeosdevicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (ChromeOsDevice) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.chromeosdevices.patch', ordered_params=[u'customerId', u'deviceId'], path_params=[u'customerId', u'deviceId'], query_params=[u'projection'], relative_path=u'customer/{customerId}/devices/chromeos/{deviceId}', request_field=u'chromeOsDevice', request_type_name=u'DirectoryChromeosdevicesPatchRequest', response_type_name=u'ChromeOsDevice', supports_download=False)

    def Update(self, request, global_params=None):
        """Update Chrome OS Device.

      Args:
        request: (DirectoryChromeosdevicesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (ChromeOsDevice) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.chromeosdevices.update', ordered_params=[u'customerId', u'deviceId'], path_params=[u'customerId', u'deviceId'], query_params=[u'projection'], relative_path=u'customer/{customerId}/devices/chromeos/{deviceId}', request_field=u'chromeOsDevice', request_type_name=u'DirectoryChromeosdevicesUpdateRequest', response_type_name=u'ChromeOsDevice', supports_download=False)