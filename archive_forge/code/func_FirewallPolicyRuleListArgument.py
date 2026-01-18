from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def FirewallPolicyRuleListArgument(required=False, plural=False, operation=None):
    return compute_flags.ResourceArgument(name='FIREWALL_POLICY', resource_name='firewall policy', completer=FirewallPoliciesCompleter, plural=plural, required=required, custom_plural='firewall policies', short_help='Short name of the firewall policy to {0}.'.format(operation), global_collection='compute.firewallPolicies')