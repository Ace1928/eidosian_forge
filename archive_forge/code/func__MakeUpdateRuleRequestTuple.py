from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeUpdateRuleRequestTuple(self, priority=None, firewall_policy=None, firewall_policy_rule=None):
    return (self._client.firewallPolicies, 'PatchRule', self._messages.ComputeFirewallPoliciesPatchRuleRequest(priority=priority, firewallPolicy=firewall_policy, firewallPolicyRule=firewall_policy_rule))