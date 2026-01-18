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
def GetRegionalTarget(client, resources, args, forwarding_rule_ref, include_target_service_attachment=False, include_regional_tcp_proxy=False):
    """Return the forwarding target for a regionally scoped request."""
    _ValidateRegionalTargetArgs(args)
    region_arg = forwarding_rule_ref.region
    project_arg = forwarding_rule_ref.project
    if args.target_pool:
        args.target_pool_region = args.target_pool_region or region_arg
        target_ref = flags.TARGET_POOL_ARG.ResolveAsResource(args, resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        target_region = target_ref.region
    elif args.target_instance:
        target_ref = flags.TARGET_INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=_GetZonesInRegionLister(['--target-instance-zone'], region_arg, client, project_arg or properties.VALUES.core.project.GetOrFail()))
        target_region = utils.ZoneNameToRegionName(target_ref.zone)
    elif getattr(args, 'target_vpn_gateway', None):
        args.target_vpn_gateway_region = args.target_vpn_gateway_region or region_arg
        target_ref = flags.TARGET_VPN_GATEWAY_ARG.ResolveAsResource(args, resources)
        target_region = target_ref.region
    elif getattr(args, 'backend_service', None):
        args.backend_service_region = args.backend_service_region or region_arg
        target_ref = flags.BACKEND_SERVICE_ARG.ResolveAsResource(args, resources)
        target_region = target_ref.region
    elif args.target_http_proxy:
        target_ref = flags.TargetHttpProxyArg().ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
        target_region = region_arg
    elif args.target_https_proxy:
        target_ref = flags.TargetHttpsProxyArg().ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
        target_region = region_arg
    elif args.target_ssl_proxy:
        target_ref = flags.TARGET_SSL_PROXY_ARG.ResolveAsResource(args, resources)
        target_region = region_arg
    elif args.target_tcp_proxy:
        target_ref = flags.TargetTcpProxyArg(allow_regional=include_regional_tcp_proxy).ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
        target_region = region_arg
    elif include_target_service_attachment and args.target_service_attachment:
        target_ref = flags.TargetServiceAttachmentArg().ResolveAsResource(args, resources)
        target_region = target_ref.region
        if target_region != region_arg or (args.target_service_attachment_region and region_arg and (args.target_service_attachment_region != region_arg)):
            raise exceptions.ArgumentError('The region of the provided service attachment must equal the [--region] of the forwarding rule.')
    else:
        raise exceptions.ArgumentError('\nFor a regional forwarding rule, exactly one of  ``--target-instance``,\n``--target-pool``, ``--target-http-proxy``, ``--target-https-proxy``,\n``--target-grpc-proxy``, ``--target-ssl-proxy``, ``--target-tcp-proxy``,\n{} ``--target-vpn-gateway`` or ``--backend-service`` must be specified.\n'.format('``--target-service-attachment``,' if include_target_service_attachment else None))
    return (target_ref, target_region)