from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddBgpIdentifierRangeArg(parser):
    """Adds BGP identifier range argument for routers."""
    parser.add_argument('--bgp-identifier-range', type=utils.IPV4RangeArgument, help='The range of valid BGP Identifiers for this Router. Must be a link-local IPv4 range from 169.254.0.0/16, of size at least /30, even if the BGP sessions are over IPv6. It must not overlap with any IPv4 BGP session ranges. This is commonly called "router ID" by other vendors.')