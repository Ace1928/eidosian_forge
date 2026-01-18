from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def ResetSslConfig(self, request, global_params=None):
    """Deletes all client certificates and generates a new server SSL certificate for the instance.

      Args:
        request: (SqlInstancesResetSslConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ResetSslConfig')
    return self._RunMethod(config, request, global_params=global_params)