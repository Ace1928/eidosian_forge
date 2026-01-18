from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class MobiledevicesService(base_api.BaseApiService):
    """Service class for the mobiledevices resource."""
    _NAME = u'mobiledevices'

    def __init__(self, client):
        super(AdminDirectoryV1.MobiledevicesService, self).__init__(client)
        self._upload_configs = {}

    def Action(self, request, global_params=None):
        """Take action on Mobile Device.

      Args:
        request: (DirectoryMobiledevicesActionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryMobiledevicesActionResponse) The response message.
      """
        config = self.GetMethodConfig('Action')
        return self._RunMethod(config, request, global_params=global_params)
    Action.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.mobiledevices.action', ordered_params=[u'customerId', u'resourceId'], path_params=[u'customerId', u'resourceId'], query_params=[], relative_path=u'customer/{customerId}/devices/mobile/{resourceId}/action', request_field=u'mobileDeviceAction', request_type_name=u'DirectoryMobiledevicesActionRequest', response_type_name=u'DirectoryMobiledevicesActionResponse', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete Mobile Device.

      Args:
        request: (DirectoryMobiledevicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryMobiledevicesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.mobiledevices.delete', ordered_params=[u'customerId', u'resourceId'], path_params=[u'customerId', u'resourceId'], query_params=[], relative_path=u'customer/{customerId}/devices/mobile/{resourceId}', request_field='', request_type_name=u'DirectoryMobiledevicesDeleteRequest', response_type_name=u'DirectoryMobiledevicesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve Mobile Device.

      Args:
        request: (DirectoryMobiledevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (MobileDevice) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.mobiledevices.get', ordered_params=[u'customerId', u'resourceId'], path_params=[u'customerId', u'resourceId'], query_params=[u'projection'], relative_path=u'customer/{customerId}/devices/mobile/{resourceId}', request_field='', request_type_name=u'DirectoryMobiledevicesGetRequest', response_type_name=u'MobileDevice', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve all Mobile Devices of a customer (paginated).

      Args:
        request: (DirectoryMobiledevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (MobileDevices) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.mobiledevices.list', ordered_params=[u'customerId'], path_params=[u'customerId'], query_params=[u'maxResults', u'orderBy', u'pageToken', u'projection', u'query', u'sortOrder'], relative_path=u'customer/{customerId}/devices/mobile', request_field='', request_type_name=u'DirectoryMobiledevicesListRequest', response_type_name=u'MobileDevices', supports_download=False)