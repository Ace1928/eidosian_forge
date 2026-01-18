from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def AttachNetworkEndpoints(self, request, global_params=None):
    """Attach a list of network endpoints to the specified network endpoint group.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsAttachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AttachNetworkEndpoints')
    return self._RunMethod(config, request, global_params=global_params)