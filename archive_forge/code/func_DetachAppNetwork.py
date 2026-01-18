from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
def DetachAppNetwork(self, request, global_params=None):
    """Detaches an app network from a SSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysDetachAppNetworkRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DetachAppNetwork')
    return self._RunMethod(config, request, global_params=global_params)