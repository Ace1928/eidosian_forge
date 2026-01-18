from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCreateExternalVpnGatewayArgs(parser):
    """Adds common arguments for creating external VPN gateways."""
    parser.add_argument('--description', help='Textual description of the External VPN Gateway.')
    parser.add_argument('--interfaces', required=True, metavar=ALLOWED_METAVAR, type=arg_parsers.ArgList(min_length=0, max_length=4), help='      Map of interfaces from interface ID to interface IP address for the External VPN Gateway.\n\n      There can be one, two, or four interfaces in the map.\n\n      For example, to create an external VPN gateway with one interface:\n\n        $ {command} MY-EXTERNAL-GATEWAY --interfaces 0=192.0.2.0\n\n      To create an external VPN gateway with two interfaces:\n        $ {command} MY-EXTERNAL-GATEWAY --interfaces 0=192.0.2.0,1=192.0.2.1\n\n      To create an external VPN gateway with four interfaces:\n        $ {command} MY-EXTERNAL-GATEWAY --interfaces 0=192.0.2.0,1=192.0.2.1,2=192.0.2.3,3=192.0.2.4\n\n      To create an external VPN gateway with IPv6 addresses on four interfaces:\n        $ {command} MY-EXTERNAL-GATEWAY --interfaces 0=2001:db8::1,1=2001:db8::2,2=2001:db8::3,3=2001:db8::4\n\n      Note that the redundancy type of the gateway will be automatically inferred based on the number\n      of interfaces provided:\n\n        1 interface: `SINGLE_IP_INTERNALLY_REDUNDANT`\n\n        2 interfaces: `TWO_IPS_REDUNDANCY`\n\n        4 interfaces: `FOUR_IPS_REDUNDANCY`\n      ')