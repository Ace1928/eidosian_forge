from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def ConvertNetworkFirewallRulesToEffectiveFwRules(network_firewalls):
    """Convert network firewall rules to effective firewall rules."""
    result = []
    for rule in network_firewalls:
        item = {}
        item.update({'type': 'network-firewall'})
        item.update({'description': rule.description})
        item.update({'priority': rule.priority})
        item.update({'direction': rule.direction})
        if rule.allowed:
            item.update({'action': 'ALLOW'})
        else:
            item.update({'action': 'DENY'})
        if rule.sourceRanges:
            item.update({'ip_ranges': rule.sourceRanges})
        if rule.destinationRanges:
            item.update({'ip_ranges': rule.destinationRanges})
        if rule.targetServiceAccounts:
            item.update({'target_svc_acct': rule.targetServiceAccounts})
        if rule.targetTags:
            item.update({'target_tags': rule.targetTags})
        if rule.sourceTags:
            item.update({'src_tags': rule.sourceTags})
        if rule.sourceServiceAccounts:
            item.update({'src_svc_acct': rule.sourceTags})
        if rule.disabled:
            item.update({'disabled': True})
        else:
            item.update({'disabled': False})
        item.update({'name': rule.name})
        result.append(item)
    return result