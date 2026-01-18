from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsListFirewallPolicy(parser):
    """Adds the argument for firewall policy list."""
    group = parser.add_group(required=True, mutex=True)
    group.add_argument('--organization', help='Organization in which firewall policies are listed')
    group.add_argument('--folder', help='Folder in which firewall policies are listed')