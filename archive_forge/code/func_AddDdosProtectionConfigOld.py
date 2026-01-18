from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddDdosProtectionConfigOld(parser, required=False):
    """Adds the cloud armor DDoS protection config arguments to the argparse."""
    parser.add_argument('--ddos-protection', choices=['STANDARD', 'ADVANCED', 'ADVANCED_PREVIEW'], type=lambda x: x.upper(), required=required, help='The DDoS protection level for network load balancing and instances with external IPs')