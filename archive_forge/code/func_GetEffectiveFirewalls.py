from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetEffectiveFirewalls(self, request, global_params=None):
    """Returns the effective firewalls on a given network.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesGetEffectiveFirewallsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponse) The response message.
      """
    config = self.GetMethodConfig('GetEffectiveFirewalls')
    return self._RunMethod(config, request, global_params=global_params)