from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def FirewallPolicyRuleArgument(required=False, plural=False, operation=None):
    return compute_flags.ResourceArgument(name='priority', resource_name='firewall policy rule', completer=FirewallPoliciesCompleter, plural=plural, required=required, global_collection='compute.firewallPolicies', short_help='Priority of the firewall policy rule to {}.'.format(operation))