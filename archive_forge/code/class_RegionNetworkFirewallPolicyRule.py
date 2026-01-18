from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class RegionNetworkFirewallPolicyRule(RegionNetworkFirewallPolicy):
    """Abstracts Region Network FirewallPolicy Rule."""

    def __init__(self, ref=None, compute_client=None):
        super(RegionNetworkFirewallPolicyRule, self).__init__(ref=ref, compute_client=compute_client)

    def _MakeCreateRuleRequestTuple(self, firewall_policy=None, firewall_policy_rule=None):
        return (self._client.regionNetworkFirewallPolicies, 'AddRule', self._messages.ComputeRegionNetworkFirewallPoliciesAddRuleRequest(firewallPolicy=firewall_policy, firewallPolicyRule=firewall_policy_rule, project=self.ref.project, region=self.ref.region))

    def _MakeDeleteRuleRequestTuple(self, priority=None, firewall_policy=None):
        return (self._client.regionNetworkFirewallPolicies, 'RemoveRule', self._messages.ComputeRegionNetworkFirewallPoliciesRemoveRuleRequest(firewallPolicy=firewall_policy, priority=priority, project=self.ref.project, region=self.ref.region))

    def _MakeDescribeRuleRequestTuple(self, priority=None, firewall_policy=None):
        return (self._client.regionNetworkFirewallPolicies, 'GetRule', self._messages.ComputeRegionNetworkFirewallPoliciesGetRuleRequest(firewallPolicy=firewall_policy, priority=priority, project=self.ref.project, region=self.ref.region))

    def _MakeUpdateRuleRequestTuple(self, priority=None, firewall_policy=None, firewall_policy_rule=None):
        return (self._client.regionNetworkFirewallPolicies, 'PatchRule', self._messages.ComputeRegionNetworkFirewallPoliciesPatchRuleRequest(priority=priority, firewallPolicy=firewall_policy, firewallPolicyRule=firewall_policy_rule, project=self.ref.project, region=self.ref.region))

    def Create(self, firewall_policy=None, firewall_policy_rule=None, only_generate_request=False):
        """Sends request to create a region network firewall policy rule."""
        requests = [self._MakeCreateRuleRequestTuple(firewall_policy=firewall_policy, firewall_policy_rule=firewall_policy_rule)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Delete(self, priority=None, firewall_policy=None, only_generate_request=False):
        """Sends request to delete a region network firewall policy rule."""
        requests = [self._MakeDeleteRuleRequestTuple(priority=priority, firewall_policy=firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Describe(self, priority=None, firewall_policy=None, only_generate_request=False):
        """Sends request to describe a region firewall policy rule."""
        requests = [self._MakeDescribeRuleRequestTuple(priority=priority, firewall_policy=firewall_policy)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests

    def Update(self, priority=None, firewall_policy=None, firewall_policy_rule=None, only_generate_request=False):
        """Sends request to update a region network firewall policy rule."""
        requests = [self._MakeUpdateRuleRequestTuple(priority=priority, firewall_policy=firewall_policy, firewall_policy_rule=firewall_policy_rule)]
        if not only_generate_request:
            return self._compute_client.MakeRequests(requests)
        return requests