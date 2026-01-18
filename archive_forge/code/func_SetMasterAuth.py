from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def SetMasterAuth(self, request, global_params=None):
    """Sets master auth materials. Currently supports changing the admin password or a specific cluster, either via password generation or explicitly setting the password.

      Args:
        request: (SetMasterAuthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetMasterAuth')
    return self._RunMethod(config, request, global_params=global_params)