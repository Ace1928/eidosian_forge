from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def Wipe(self, request, global_params=None):
    """Wipes all data on the specified device.

      Args:
        request: (CloudidentityDevicesWipeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Wipe')
    return self._RunMethod(config, request, global_params=global_params)