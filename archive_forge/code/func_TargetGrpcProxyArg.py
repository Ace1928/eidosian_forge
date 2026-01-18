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
def TargetGrpcProxyArg():
    """Return a resource argument for parsing a target gRPC proxy."""
    target_grpc_proxy_arg = compute_flags.ResourceArgument(name='--target-grpc-proxy', required=False, resource_name='target gRPC proxy', global_collection='compute.targetGrpcProxies', short_help='Target gRPC proxy that receives the traffic.', detailed_help='Target gRPC proxy that receives the traffic.', region_explanation=None)
    return target_grpc_proxy_arg