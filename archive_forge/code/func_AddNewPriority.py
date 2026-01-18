from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddNewPriority(parser, operation=None):
    """Adds the new firewall policy rule priority to the argparse."""
    parser.add_argument('--new-priority', help='New priority for the rule to {}. Valid in [0, 65535]. '.format(operation))