from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRulePriority(parser, operation=None):
    """Adds the rule priority argument to the argparse."""
    parser.add_argument('priority', help='Priority of the rule to be {}. Valid in [0, 65535].'.format(operation))