from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddInternalIpv6AddressArg(parser):
    parser.add_argument('--internal-ipv6-address', type=str, help='\n        Assigns the given internal IPv6 address or range to an instance.\n        The address must be the first IP address in the range or a /96 IP\n        address range. This option can only be used on a dual stack instance\n        network interface.\n      ')