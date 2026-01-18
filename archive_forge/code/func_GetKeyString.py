from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apikeys.v2 import apikeys_v2_messages as messages
def GetKeyString(self, request, global_params=None):
    """Get the key string for an API key. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysGetKeyStringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2GetKeyStringResponse) The response message.
      """
    config = self.GetMethodConfig('GetKeyString')
    return self._RunMethod(config, request, global_params=global_params)