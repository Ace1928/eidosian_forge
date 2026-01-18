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
def AddIpAddressSelectionPolicy(parser):
    parser.add_argument('--ip-address-selection-policy', choices=['IPV4_ONLY', 'PREFER_IPV6', 'IPV6_ONLY'], type=lambda x: x.replace('-', '_').upper(), help="      Specifies a preference for traffic sent from the proxy to the backend (or\n      from the client to the backend for proxyless gRPC).\n\n      Can only be set if load balancing scheme is INTERNAL_SELF_MANAGED,\n      INTERNAL_MANAGED or EXTERNAL_MANAGED.\n\n      The possible values are:\n\n       IPV4_ONLY\n         Only send IPv4 traffic to the backends of the backend service,\n         regardless of traffic from the client to the proxy. Only IPv4\n         health checks are used to check the health of the backends.\n\n       PREFER_IPV6\n         Prioritize the connection to the endpoint's IPv6 address over its IPv4\n         address (provided there is a healthy IPv6 address).\n\n       IPV6_ONLY\n         Only send IPv6 traffic to the backends of the backend service,\n         regardless of traffic from the client to the proxy. Only IPv6\n         health checks are used to check the health of the backends.\n      ")