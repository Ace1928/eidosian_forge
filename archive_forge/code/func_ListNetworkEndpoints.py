from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListNetworkEndpoints(self, request, global_params=None):
    """Lists the network endpoints in the specified network endpoint group.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsListNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupsListNetworkEndpoints) The response message.
      """
    config = self.GetMethodConfig('ListNetworkEndpoints')
    return self._RunMethod(config, request, global_params=global_params)