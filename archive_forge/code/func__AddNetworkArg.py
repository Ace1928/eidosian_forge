from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddNetworkArg(parser):
    parser.add_argument('--network', required=True, help='        Network for this packet mirroring.\n        Only the packets in this network will be mirrored. It is mandatory\n        that all mirrored VMs have a network interface controller (NIC) in\n        the given network. All mirrored subnetworks should belong to the\n        given network.\n\n        You can provide this as the full URL to the network, partial URL,\n        or name.\n        For example, the following are valid values:\n          * https://compute.googleapis.com/compute/v1/projects/myproject/\n            global/networks/network-1\n          * projects/myproject/global/networks/network-1\n          * network-1\n        ')