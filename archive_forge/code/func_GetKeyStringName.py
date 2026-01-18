from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apikeys.v2alpha1 import apikeys_v2alpha1_messages as messages
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