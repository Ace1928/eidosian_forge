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
def AddDropTrafficIfUnhealthy(parser, default):
    """Adds the drop traffic if unhealthy argument to the argparse."""
    parser.add_argument('--drop-traffic-if-unhealthy', action='store_true', default=default, help='      Applicable only for backend service-based external and internal\n      passthrough Network Load Balancers as part of a connection tracking\n      policy. Not applicable to any other load balancer. This option instructs\n      the load balancer to drop packets when all instances or endpoints in\n      primary and failover backends do not pass their load balancer health\n      checks. For details, see: [Dropping traffic when all backend VMs are\n      unhealthy for internal passthrough Network Load\n      Balancers](https://cloud.google.com/load-balancing/docs/internal/failover-overview#drop_traffic)\n      and [Dropping traffic when all backend VMs are unhealthy for external\n      passthrough Network Load\n      Balancers](https://cloud.google.com/load-balancing/docs/network/networklb-failover-overview#drop_traffic).\n      ')