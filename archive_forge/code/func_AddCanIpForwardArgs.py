from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCanIpForwardArgs(parser):
    parser.add_argument('--can-ip-forward', action='store_true', help='        If provided, allows the VMs created from the imported machine\n        image to send and receive packets with non-matching destination or\n        source IP addresses.\n        ')