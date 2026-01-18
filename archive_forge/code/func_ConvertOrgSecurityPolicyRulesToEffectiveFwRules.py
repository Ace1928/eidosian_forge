from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def ConvertOrgSecurityPolicyRulesToEffectiveFwRules(security_policy):
    """Convert organization security policy rules to effective firewall rules."""
    result = []
    for rule in security_policy.rules:
        item = {}
        item.update({'type': 'org-firewall'})
        item.update({'description': rule.description})
        item.update({'firewall_policy_name': security_policy.id})
        item.update({'priority': rule.priority})
        item.update({'direction': rule.direction})
        item.update({'action': rule.action.upper()})
        item.update({'disabled': 'False'})
        if rule.match.config.srcIpRanges:
            item.update({'ip_ranges': rule.match.config.srcIpRanges})
        if rule.match.config.destIpRanges:
            item.update({'ip_ranges': rule.match.config.destIpRanges})
        if rule.targetServiceAccounts:
            item.update({'target_svc_acct': rule.targetServiceAccounts})
        if rule.targetResources:
            item.update({'target_resources': rule.targetResources})
        result.append(item)
    return result