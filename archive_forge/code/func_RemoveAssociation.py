from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def RemoveAssociation(self, request, global_params=None):
    """Removes an association for the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesRemoveAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RemoveAssociation')
    return self._RunMethod(config, request, global_params=global_params)