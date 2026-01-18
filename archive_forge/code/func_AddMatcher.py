from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMatcher(parser, required=True):
    """Adds the matcher arguments to the argparse."""
    matcher = parser.add_group(mutex=True, required=required, help='Security policy rule matcher.')
    matcher.add_argument('--src-ip-ranges', type=arg_parsers.ArgList(), metavar='SRC_IP_RANGE', help='The source IPs/IP ranges to match for this rule. To match all IPs specify *.')
    matcher.add_argument('--expression', help='The Cloud Armor rules language expression to match for this rule.')