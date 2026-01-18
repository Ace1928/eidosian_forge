from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddConnectionTrackingPolicy(parser):
    """Add flags related to connection tracking policy.

  Args:
    parser: The parser that parses args from user input.
  """
    parser.add_argument('--connection-persistence-on-unhealthy-backends', choices=['DEFAULT_FOR_PROTOCOL', 'NEVER_PERSIST', 'ALWAYS_PERSIST'], type=lambda x: x.replace('-', '_').upper(), default=None, help='      Specifies connection persistence when backends are unhealthy.\n      The default value is DEFAULT_FOR_PROTOCOL.\n      ')
    parser.add_argument('--tracking-mode', choices=['PER_CONNECTION', 'PER_SESSION'], type=lambda x: x.replace('-', '_').upper(), default=None, help='      Specifies the connection key used for connection tracking.\n      The default value is PER_CONNECTION. Applicable only for backend\n      service-based external and internal passthrough Network Load\n      Balancers as part of a connection tracking policy.\n      For details, see: [Connection tracking mode for\n      internal passthrough Network Load Balancers\n      balancing](https://cloud.google.com/load-balancing/docs/internal#tracking-mode)\n      and [Connection tracking mode for external passthrough Network Load\n      Balancers](https://cloud.google.com/load-balancing/docs/network/networklb-backend-service#tracking-mode).\n      ')
    parser.add_argument('--idle-timeout-sec', type=arg_parsers.Duration(), default=None, help='      Specifies how long to keep a connection tracking table entry while there\n      is no matching traffic (in seconds). Applicable only for backend\n      service-based external and internal passthrough Network Load\n      Balancers as part of a connection tracking policy.\n      ')
    parser.add_argument('--enable-strong-affinity', action=arg_parsers.StoreTrueFalseAction, help='      Enable or disable strong session affinity.\n      This is only available for loadbalancingScheme EXTERNAL.\n      ')