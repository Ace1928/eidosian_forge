from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMtuArg(parser):
    """Adds the --mtu flag."""
    parser.add_argument('--mtu', type=int, help='Maximum transmission unit (MTU) is the size of the largest\n              IP packet that can be transmitted on this network. Default value\n              is 1460 bytes. The minimum value is 1300 bytes and the maximum\n              value is 8896 bytes. The MTU advertised via DHCP to all instances\n              attached to this network.')