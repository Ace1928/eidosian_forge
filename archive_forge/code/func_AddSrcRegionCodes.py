from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSrcRegionCodes(parser):
    """Adds a source region code to this rule."""
    parser.add_argument('--src-region-codes', type=arg_parsers.ArgList(), metavar='SOURCE_REGION_CODES', required=False, help='Source Region Code to match for this rule. Can only be specified if DIRECTION is `ingress`.')