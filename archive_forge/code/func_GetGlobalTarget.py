from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import properties
def GetGlobalTarget(resources, args):
    """Return the forwarding target for a globally scoped request."""
    _ValidateGlobalTargetArgs(args)
    if args.target_http_proxy:
        return flags.TargetHttpProxyArg().ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
    if args.target_https_proxy:
        return flags.TargetHttpsProxyArg().ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
    if args.target_grpc_proxy:
        return flags.TargetGrpcProxyArg().ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
    if args.target_ssl_proxy:
        return flags.TARGET_SSL_PROXY_ARG.ResolveAsResource(args, resources)
    if getattr(args, 'target_tcp_proxy', None):
        return flags.TARGET_TCP_PROXY_ARG.ResolveAsResource(args, resources)