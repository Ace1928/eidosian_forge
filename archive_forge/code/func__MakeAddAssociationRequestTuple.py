from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeAddAssociationRequestTuple(self, association, firewall_policy_id, replace_existing_association):
    return (self._client.firewallPolicies, 'AddAssociation', self._messages.ComputeFirewallPoliciesAddAssociationRequest(firewallPolicyAssociation=association, firewallPolicy=firewall_policy_id, replaceExistingAssociation=replace_existing_association))