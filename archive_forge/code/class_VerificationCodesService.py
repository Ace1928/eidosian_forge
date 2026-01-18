from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class VerificationCodesService(base_api.BaseApiService):
    """Service class for the verificationCodes resource."""
    _NAME = u'verificationCodes'

    def __init__(self, client):
        super(AdminDirectoryV1.VerificationCodesService, self).__init__(client)
        self._upload_configs = {}

    def Generate(self, request, global_params=None):
        """Generate new backup verification codes for the user.

      Args:
        request: (DirectoryVerificationCodesGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryVerificationCodesGenerateResponse) The response message.
      """
        config = self.GetMethodConfig('Generate')
        return self._RunMethod(config, request, global_params=global_params)
    Generate.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.verificationCodes.generate', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/verificationCodes/generate', request_field='', request_type_name=u'DirectoryVerificationCodesGenerateRequest', response_type_name=u'DirectoryVerificationCodesGenerateResponse', supports_download=False)

    def Invalidate(self, request, global_params=None):
        """Invalidate the current backup verification codes for the user.

      Args:
        request: (DirectoryVerificationCodesInvalidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryVerificationCodesInvalidateResponse) The response message.
      """
        config = self.GetMethodConfig('Invalidate')
        return self._RunMethod(config, request, global_params=global_params)
    Invalidate.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.verificationCodes.invalidate', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/verificationCodes/invalidate', request_field='', request_type_name=u'DirectoryVerificationCodesInvalidateRequest', response_type_name=u'DirectoryVerificationCodesInvalidateResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the current set of valid backup verification codes for the specified user.

      Args:
        request: (DirectoryVerificationCodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (VerificationCodes) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.verificationCodes.list', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/verificationCodes', request_field='', request_type_name=u'DirectoryVerificationCodesListRequest', response_type_name=u'VerificationCodes', supports_download=False)