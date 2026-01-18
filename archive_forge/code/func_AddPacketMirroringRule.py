from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def AddPacketMirroringRule(self, request, global_params=None):
    """Inserts a packet mirroring rule into a firewall policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesAddPacketMirroringRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AddPacketMirroringRule')
    return self._RunMethod(config, request, global_params=global_params)