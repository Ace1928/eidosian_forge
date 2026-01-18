from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class AspsService(base_api.BaseApiService):
    """Service class for the asps resource."""
    _NAME = u'asps'

    def __init__(self, client):
        super(AdminDirectoryV1.AspsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete an ASP issued by a user.

      Args:
        request: (DirectoryAspsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryAspsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.asps.delete', ordered_params=[u'userKey', u'codeId'], path_params=[u'codeId', u'userKey'], query_params=[], relative_path=u'users/{userKey}/asps/{codeId}', request_field='', request_type_name=u'DirectoryAspsDeleteRequest', response_type_name=u'DirectoryAspsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Get information about an ASP issued by a user.

      Args:
        request: (DirectoryAspsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Asp) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.asps.get', ordered_params=[u'userKey', u'codeId'], path_params=[u'codeId', u'userKey'], query_params=[], relative_path=u'users/{userKey}/asps/{codeId}', request_field='', request_type_name=u'DirectoryAspsGetRequest', response_type_name=u'Asp', supports_download=False)

    def List(self, request, global_params=None):
        """List the ASPs issued by a user.

      Args:
        request: (DirectoryAspsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Asps) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.asps.list', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/asps', request_field='', request_type_name=u'DirectoryAspsListRequest', response_type_name=u'Asps', supports_download=False)