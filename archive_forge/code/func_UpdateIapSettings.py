from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
def UpdateIapSettings(self, request, global_params=None):
    """Updates the IAP settings on a particular IAP protected resource. It replaces all fields unless the `update_mask` is set.

      Args:
        request: (IapUpdateIapSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IapSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateIapSettings')
    return self._RunMethod(config, request, global_params=global_params)