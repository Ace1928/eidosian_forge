from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def RegionalWebhook(self, request, global_params=None):
    """ReceiveRegionalWebhook is called when the API receives a regional GitHub webhook.

      Args:
        request: (CloudbuildLocationsRegionalWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('RegionalWebhook')
    return self._RunMethod(config, request, global_params=global_params)