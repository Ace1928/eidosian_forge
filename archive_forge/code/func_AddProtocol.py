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
def AddProtocol(parser, default='HTTP', support_unspecified_protocol=False):
    """Adds --protocol flag to the argparse.

  Args:
    parser: An argparse.ArgumentParser instance.
    default: The default protocol if this flag is unspecified.
    support_unspecified_protocol: Indicates if UNSPECIFIED is a valid protocol.
  """
    ilb_protocols = 'TCP, UDP, UNSPECIFIED' if support_unspecified_protocol else 'TCP, UDP'
    td_protocols = 'HTTP, HTTPS, HTTP2, GRPC'
    netlb_protocols = 'TCP, UDP, UNSPECIFIED' if support_unspecified_protocol else 'TCP, UDP'
    parser.add_argument('--protocol', default=default, type=lambda x: x.upper(), help='      Protocol for incoming requests.\n\n      If the `load-balancing-scheme` is `INTERNAL` (Internal passthrough\n      Network Load Balancer), the protocol must be one of: {0}.\n\n      If the `load-balancing-scheme` is `INTERNAL_SELF_MANAGED` (Traffic\n      Director), the protocol must be one of: {1}.\n\n      If the `load-balancing-scheme` is `INTERNAL_MANAGED` (Internal Application\n      Load Balancer), the protocol must be one of: HTTP, HTTPS, HTTP2.\n\n      If the `load-balancing-scheme` is `EXTERNAL` and `region` is not set\n      (Classic Application Load Balancer and global external proxy Network\n      Load Balancer), the protocol must be one of: HTTP, HTTPS, HTTP2, SSL, TCP.\n\n      If the `load-balancing-scheme` is `EXTERNAL` and `region` is set\n      (External passthrough Network Load Balancer), the protocol must be one\n      of: {2}.\n\n      If the `load-balancing-scheme` is `EXTERNAL_MANAGED` (Envoy based\n      Global and regional external Application Load Balancers), the protocol\n      must be one of: HTTP, HTTPS, HTTP2.\n      '.format(ilb_protocols, td_protocols, netlb_protocols))