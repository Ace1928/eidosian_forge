from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def Approve(self, request, global_params=None):
    """Approves device to access user data.

      Args:
        request: (CloudidentityDevicesDeviceUsersApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Approve')
    return self._RunMethod(config, request, global_params=global_params)