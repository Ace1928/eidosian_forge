from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSrcIpRanges(parser, required=False):
    """Adds the source IP ranges."""
    parser.add_argument('--src-ip-ranges', type=arg_parsers.ArgList(), required=required, metavar='SRC_IP_RANGE', help='Source IP ranges to match for this rule.')