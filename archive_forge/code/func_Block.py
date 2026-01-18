from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def Block(self, request, global_params=None):
    """Blocks device from accessing user data.

      Args:
        request: (CloudidentityDevicesDeviceUsersBlockRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Block')
    return self._RunMethod(config, request, global_params=global_params)