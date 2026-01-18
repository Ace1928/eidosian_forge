from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def FirewallRuleArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='firewall rule', completer=FirewallsCompleter, plural=plural, required=required, global_collection='compute.firewalls')