from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddReplaceCustomLearnedRoutesArgs(parser):
    """Adds common arguments for replacing custom learned routes.

  Args:
    parser: The parser to parse arguments.
  """
    parser.add_argument('--custom-learned-route-priority', type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=65535), metavar='PRIORITY', help='An integral value `0` <= priority <= `65535`, to be applied to all\n              custom learned route IP address ranges for this peer. If not\n              specified, a Google-managed priority value of 100 is used. The\n              routes with the lowest priority value win.')
    parser.add_argument('--set-custom-learned-route-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='The list of user-defined custom learned route IP address ranges\n              for this peer. This list is a comma separated IP address ranges\n              such as `1.2.3.4`,`6.7.0.0/16`,`2001:db8:abcd:12::/64` where each\n              IP address range must be a valid CIDR-formatted prefix. If an IP\n              address is provided without a subnet mask, it is interpreted as a\n              /32 singular IP address range for IPv4, and /128 for IPv6.')