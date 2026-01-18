from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddDestRegionCodes(parser):
    """Adds a destination region code to this rule."""
    parser.add_argument('--dest-region-codes', type=arg_parsers.ArgList(), metavar='DEST_REGION_CODES', required=False, help='Destination Region Code to match for this rule. Can only be specified if DIRECTION is `egress`.')