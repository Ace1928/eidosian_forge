from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeDescribeRuleRequestTuple(self, priority=None, firewall_policy=None):
    return (self._client.firewallPolicies, 'GetRule', self._messages.ComputeFirewallPoliciesGetRuleRequest(firewallPolicy=firewall_policy, priority=priority))