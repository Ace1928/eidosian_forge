from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMatchArg(parser, required=False):
    """Adds common arguments for creating and updating NAT Rules."""
    help_text = textwrap.dedent('\n      CEL Expression used to identify traffic to which this rule applies.\n\n      * Supported attributes (Public NAT): destination.ip\n      * Supported attributes (Private NAT): nexthop.hub\n      * Supported methods (Public Nat): inIpRange\n      * Supported operators (Public NAT): ||, ==\n      * Supported operators (Private NAT): ==\n\n      Examples of allowed Match expressions (Public NAT):\n      * \'inIpRange(destination.ip, "203.0.113.0/24")\'\'\n      * \'destination.ip == "203.0.113.7"\'\n      * \'destination.ip == "203.0.113.7" || inIpRange(destination.ip, "203.0.113.16/25")\'\n\n      Example of allowed Match expression (Private NAT):\n      * nexthop.hub == "//networkconnectivity.googleapis.com/projects/p1/locations/global/hubs/h1"\n  ')
    parser.add_argument('--match', help=help_text, required=required)