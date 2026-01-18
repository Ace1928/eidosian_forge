from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import arg_parsers
def AddInterfaceArgs(parser, for_update=False):
    """Adds common arguments for routers add-interface or update-interface."""
    help_text = 'The argument group for configuring the interface for the router.'
    operation = 'added'
    if for_update:
        operation = 'updated'
    parser.add_argument('--interface-name', help='The name of the interface being {0}.'.format(operation), required=True)
    interface_group = parser.add_argument_group(mutex=True, help=help_text, required=True)
    southbound_interface_group = interface_group.add_argument_group(help='The argument group for adding southbound interfaces to edge router.')
    southbound_interface_group.add_argument('--subnetwork', help='Subnetwork of the interface being {0}.'.format(operation))
    northbound_interface_group = interface_group.add_argument_group(help='The argument group for adding northbound interfaces to edge router.')
    northbound_interface_group.add_argument('--interconnect-attachment', help='Interconnect attachment of the interface being {0}.'.format(operation))
    northbound_interface_group.add_argument('--ip-address', type=utils.IPArgument, help='Link-local address of the router for this interface.')
    northbound_interface_group.add_argument('--ip-mask-length', type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=128), help='Subnet mask for the link-local IP range of the interface. The interface IP address and BGP peer IP address must be selected from the subnet defined by this link-local range.')
    loopback_interface_group = interface_group.add_argument_group(help='The argument group for adding loopback interfaces to edge router.')
    loopback_interface_group.add_argument('--loopback-ip-addresses', type=arg_parsers.ArgList(), metavar='LOOPBACK_IP_ADDRESSES', help='The list of ip ranges for the loopback interface.')