from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
class OrgFirewallPolicyRule(OrgFirewallPolicy):
    """Abstracts Organization FirewallPolicy Rule."""

    def __init__(self, ref=None, compute_client=None, resources=None, version='beta'):
        super(OrgFirewallPolicyRule, self).__init__(ref=ref, compute_client=compute_client, resources=resources, version=version)

    def _MakeCreateRuleRequestTuple(self, firewall_policy=None, firewall_policy_rule=None):
        return (self._client.firewallPolicies, 'AddRule', self._messages.ComputeFirewallPoliciesAddRuleRequest(firewallPolicy=firewall_policy, firewallPolicyRule=firewall_policy_rule))

    def _MakeDeleteRuleRequestTuple(self, priority=None, firewall_policy=None):
        return (self._client.firewallPolicies, 'RemoveRule', self._messages.ComputeFirewallPoliciesRemoveRuleRequest(firewallPolicy=firewall_policy, priority=priority))

    def _MakeDescribeRuleRequestTuple(self, priority=None, firewall_policy=None):
        return (self._client.firewallPolicies, 'GetRule', self._messages.ComputeFirewallPoliciesGetRuleRequest(firewallPolicy=firewall_policy, priority=priority))

    def _MakeUpdateRuleRequestTuple(self, priority=None, firewall_policy=None, firewall_policy_rule=None):
        return (self._client.firewallPolicies, 'PatchRule', self._messages.ComputeFirewallPoliciesPatchRuleRequest(priority=priority, firewallPolicy=firewall_policy, firewallPolicyRule=firewall_policy_rule))

    def Create(self, firewall_policy=None, firewall_policy_rule=None, batch_mode=False, only_generate_request=False):
        """Sends request to create an organization firewall policy rule."""
        if batch_mode:
            requests = [self._MakeCreateRuleRequestTuple(firewall_policy=firewall_policy, firewall_policy_rule=firewall_policy_rule)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.AddRule(self._MakeCreateRuleRequestTuple(firewall_policy=firewall_policy, firewall_policy_rule=firewall_policy_rule)[2])
        return self.WaitOperation(op_res, message='Adding a rule to the organization firewall policy.')

    def Delete(self, priority=None, firewall_policy_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to delete an organization firewall policy rule."""
        if batch_mode:
            requests = [self._MakeDeleteRuleRequestTuple(priority=priority, firewall_policy=firewall_policy_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.RemoveRule(self._MakeDeleteRuleRequestTuple(priority=priority, firewall_policy=firewall_policy_id)[2])
        return self.WaitOperation(op_res, message='Deleting a rule from the organization firewall policy.')

    def Describe(self, priority=None, firewall_policy_id=None, batch_mode=False, only_generate_request=False):
        """Sends request to describe a firewall policy rule."""
        if batch_mode:
            requests = [self._MakeDescribeRuleRequestTuple(priority=priority, firewall_policy=firewall_policy_id)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        return self._service.GetRule(self._MakeDescribeRuleRequestTuple(priority=priority, firewall_policy=firewall_policy_id)[2])

    def Update(self, priority=None, firewall_policy=None, firewall_policy_rule=None, batch_mode=False, only_generate_request=False):
        """Sends request to update an organization firewall policy rule."""
        if batch_mode:
            requests = [self._MakeUpdateRuleRequestTuple(priority=priority, firewall_policy=firewall_policy, firewall_policy_rule=firewall_policy_rule)]
            if not only_generate_request:
                return self._compute_client.MakeRequests(requests)
            return requests
        op_res = self._service.PatchRule(self._MakeUpdateRuleRequestTuple(priority=priority, firewall_policy=firewall_policy, firewall_policy_rule=firewall_policy_rule)[2])
        return self.WaitOperation(op_res, message='Updating a rule in the organization firewall policy.')