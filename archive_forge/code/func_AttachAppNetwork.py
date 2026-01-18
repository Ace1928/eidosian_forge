from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
def AttachAppNetwork(self, request, global_params=None):
    """Attaches an app network to a SSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysAttachAppNetworkRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AttachAppNetwork')
    return self._RunMethod(config, request, global_params=global_params)