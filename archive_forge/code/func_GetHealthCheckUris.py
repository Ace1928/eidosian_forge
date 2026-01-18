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
def GetHealthCheckUris(args, resource_resolver, resource_parser):
    """Returns health check URIs from arguments."""
    health_check_refs = []
    if args.http_health_checks:
        health_check_refs.extend(resource_resolver.HTTP_HEALTH_CHECK_ARG.ResolveAsResource(args, resource_parser))
    if getattr(args, 'https_health_checks', None):
        health_check_refs.extend(resource_resolver.HTTPS_HEALTH_CHECK_ARG.ResolveAsResource(args, resource_parser))
    if getattr(args, 'health_checks', None):
        if health_check_refs:
            raise compute_exceptions.ArgumentError('Mixing --health-checks with --http-health-checks or with --https-health-checks is not supported.')
        else:
            health_check_refs.extend(resource_resolver.HEALTH_CHECK_ARG.ResolveAsResource(args, resource_parser, default_scope=compute_scope.ScopeEnum.GLOBAL))
    if health_check_refs and getattr(args, 'no_health_checks', None):
        raise compute_exceptions.ArgumentError('Combining --health-checks, --http-health-checks, or --https-health-checks with --no-health-checks is not supported.')
    return [health_check_ref.SelfLink() for health_check_ref in health_check_refs]