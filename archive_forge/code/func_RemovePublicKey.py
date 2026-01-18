from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudshell.v1 import cloudshell_v1_messages as messages
def RemovePublicKey(self, request, global_params=None):
    """Removes a public SSH key from an environment. Clients will no longer be able to connect to the environment using the corresponding private key. If a key with the same content is not present, this will error with NOT_FOUND.

      Args:
        request: (CloudshellUsersEnvironmentsRemovePublicKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RemovePublicKey')
    return self._RunMethod(config, request, global_params=global_params)