from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddFirewallPolicyId(parser, required=True, operation=None):
    """Adds the firewall policy ID argument to the argparse."""
    parser.add_argument('--firewall-policy', required=required, help='Short name of the firewall policy into which the rule should be {}.'.format(operation))