from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def ConvertFirewallPolicyRulesToEffectiveFwRules(client, firewall_policy, support_network_firewall_policy, support_region_network_firewall_policy=True):
    """Convert organization firewall policy rules to effective firewall rules."""
    result = []
    for rule in firewall_policy.rules:
        item = {}
        if firewall_policy.type == client.messages.NetworksGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.HIERARCHY or firewall_policy.type == client.messages.InstancesGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.HIERARCHY or (support_region_network_firewall_policy and firewall_policy.type == client.messages.RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.HIERARCHY):
            item.update({'type': 'org-firewall'})
        elif support_network_firewall_policy and (firewall_policy.type == client.messages.NetworksGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.NETWORK or firewall_policy.type == client.messages.InstancesGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.NETWORK or (support_region_network_firewall_policy and firewall_policy.type == client.messages.RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.NETWORK)):
            item.update({'type': 'network-firewall-policy'})
        elif support_network_firewall_policy and (firewall_policy.type == client.messages.InstancesGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.NETWORK_REGIONAL or (support_region_network_firewall_policy and firewall_policy.type == client.messages.RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponseEffectiveFirewallPolicy.TypeValueValuesEnum.NETWORK_REGIONAL)):
            item.update({'type': 'network-regional-firewall-policy'})
        else:
            item.update({'type': 'unknown'})
        item.update({'description': rule.description})
        item.update({'firewall_policy_name': firewall_policy.name})
        item.update({'priority': rule.priority})
        item.update({'direction': rule.direction})
        item.update({'action': rule.action.upper()})
        item.update({'disabled': bool(rule.disabled)})
        if rule.match.srcIpRanges:
            item.update({'ip_ranges': rule.match.srcIpRanges})
        if rule.match.destIpRanges:
            item.update({'ip_ranges': rule.match.destIpRanges})
        if rule.targetServiceAccounts:
            item.update({'target_svc_acct': rule.targetServiceAccounts})
        if rule.targetResources:
            item.update({'target_resources': rule.targetResources})
        result.append(item)
    return result