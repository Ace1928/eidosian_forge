from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def TargetHttpProxyArg():
    """Return a resource argument for parsing a target http proxy."""
    target_http_proxy_arg = compute_flags.ResourceArgument(name='--target-http-proxy', required=False, resource_name='http proxy', global_collection='compute.targetHttpProxies', regional_collection='compute.regionTargetHttpProxies', short_help='Target HTTP proxy that receives the traffic.', detailed_help=textwrap.dedent('      Target HTTP proxy that receives the traffic. For the acceptable ports, see [Port specifications](https://cloud.google.com/load-balancing/docs/forwarding-rule-concepts#port_specifications).\n      '), region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)
    return target_http_proxy_arg