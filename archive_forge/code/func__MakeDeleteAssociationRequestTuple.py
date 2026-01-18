from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeDeleteAssociationRequestTuple(self, firewall_policy_id):
    return (self._client.firewallPolicies, 'RemoveAssociation', self._messages.ComputeFirewallPoliciesRemoveAssociationRequest(name=self.ref.Name(), firewallPolicy=firewall_policy_id))