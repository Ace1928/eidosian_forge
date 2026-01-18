from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddPrivateNetworkIpArgs(parser):
    """Set arguments for choosing the network IP address."""
    parser.add_argument('--private-network-ip', help='        Specifies the RFC1918 IP to assign to the VMs created from the\n        imported machine image. The IP should be in the subnet or legacy network\n        IP range.\n      ')