from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def NetworkSrcFirewallPolicyRuleArgument(required=False, plural=False):
    return compute_flags.ResourceArgument(name='--source-firewall-policy', resource_name='firewall policy', plural=plural, required=required, short_help='Source Firewall policy NAME with which to clone rule.', global_collection='compute.networkFirewallPolicies', regional_collection='compute.regionNetworkFirewallPolicies')