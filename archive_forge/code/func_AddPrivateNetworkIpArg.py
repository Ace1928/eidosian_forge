from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPrivateNetworkIpArg(parser, add_network_interface=False):
    """Adds --private-network-ip argument to the parser."""
    if add_network_interface:
        help_text = '\n        Specifies the RFC1918 IP to assign to the network interface. The IP\n        should be in the subnet IP range.\n      '
    else:
        help_text = '\n        Assign the given IP address to the interface. Can be specified only\n        together with --network and/or --subnetwork to choose the IP address\n        in the new subnetwork. If unspecified, then the previous IP address\n        will be allocated in the new subnetwork. If the previous IP address is\n        not available in the new subnetwork, then another available IP address\n        will be allocated automatically from the new subnetwork CIDR range.\n      '
    parser.add_argument('--private-network-ip', dest='private_network_ip', type=str, help=help_text)