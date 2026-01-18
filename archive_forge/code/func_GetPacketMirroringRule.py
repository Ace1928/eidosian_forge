from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def GetPacketMirroringRule(self, request, global_params=None):
    """Gets a packet mirroring rule of the specified priority.

      Args:
        request: (ComputeNetworkFirewallPoliciesGetPacketMirroringRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyRule) The response message.
      """
    config = self.GetMethodConfig('GetPacketMirroringRule')
    return self._RunMethod(config, request, global_params=global_params)