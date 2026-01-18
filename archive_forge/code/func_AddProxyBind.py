from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddProxyBind(parser, default):
    """Adds the proxy bind argument."""
    parser.add_argument('--proxy-bind', action='store_true', default=default, help="      This flag applies when the load_balancing_scheme of the associated\n      backend service is INTERNAL_SELF_MANAGED. When specified, the envoy binds\n      to the forwarding rule's IP address and port. By default,\n      this flag is off.\n      ")