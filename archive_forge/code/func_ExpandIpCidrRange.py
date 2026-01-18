from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ExpandIpCidrRange(self, request, global_params=None):
    """Expands the IP CIDR range of the subnetwork to a specified value.

      Args:
        request: (ComputeSubnetworksExpandIpCidrRangeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ExpandIpCidrRange')
    return self._RunMethod(config, request, global_params=global_params)