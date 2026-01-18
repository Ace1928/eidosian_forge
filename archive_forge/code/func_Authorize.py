from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudshell.v1 import cloudshell_v1_messages as messages
def Authorize(self, request, global_params=None):
    """Sends OAuth credentials to a running environment on behalf of a user. When this completes, the environment will be authorized to run various Google Cloud command line tools without requiring the user to manually authenticate.

      Args:
        request: (CloudshellUsersEnvironmentsAuthorizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Authorize')
    return self._RunMethod(config, request, global_params=global_params)