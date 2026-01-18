from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def ListIpAddresses(self, request, global_params=None):
    """Lists the internal IP addresses in the specified network.

      Args:
        request: (ComputeNetworksListIpAddressesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IpAddressesList) The response message.
      """
    config = self.GetMethodConfig('ListIpAddresses')
    return self._RunMethod(config, request, global_params=global_params)