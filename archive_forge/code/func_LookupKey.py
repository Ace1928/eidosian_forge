from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apikeys.v2 import apikeys_v2_messages as messages
def LookupKey(self, request, global_params=None):
    """Find the parent project and resource name of the API key that matches the key string in the request. If the API key has been purged, resource name will not be set. The service account must have the `apikeys.keys.lookup` permission on the parent project.

      Args:
        request: (ApikeysKeysLookupKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2LookupKeyResponse) The response message.
      """
    config = self.GetMethodConfig('LookupKey')
    return self._RunMethod(config, request, global_params=global_params)