from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apikeys.v2alpha1 import apikeys_v2alpha1_messages as messages
class V2alpha1Service(base_api.BaseApiService):
    """Service class for the v2alpha1 resource."""
    _NAME = 'v2alpha1'

    def __init__(self, client):
        super(ApikeysV2alpha1.V2alpha1Service, self).__init__(client)
        self._upload_configs = {}

    def GetKeyStringName(self, request, global_params=None):
        """Get parent and name of the Api Key which has the key string. Permission `apikeys.keys.getKeyStringName` is required on the parent.

      Args:
        request: (ApikeysGetKeyStringNameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2alpha1GetKeyStringNameResponse) The response message.
      """
        config = self.GetMethodConfig('GetKeyStringName')
        return self._RunMethod(config, request, global_params=global_params)
    GetKeyStringName.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='apikeys.getKeyStringName', ordered_params=[], path_params=[], query_params=['keyString'], relative_path='v2alpha1/keyStringName', request_field='', request_type_name='ApikeysGetKeyStringNameRequest', response_type_name='V2alpha1GetKeyStringNameResponse', supports_download=False)