from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import subnets_utils
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _CreateSubnetwork(messages, subnet_ref, network_ref, args, include_alpha_logging, include_aggregate_purpose, include_l2, include_external_ipv6_prefix):
    """Create the subnet resource."""
    subnetwork = messages.Subnetwork(name=subnet_ref.Name(), description=args.description, network=network_ref.SelfLink(), privateIpGoogleAccess=args.enable_private_ip_google_access)
    if args.range:
        subnetwork.ipCidrRange = args.range
    if args.enable_flow_logs is not None or args.logging_aggregation_interval is not None or args.logging_flow_sampling is not None or (args.logging_metadata is not None) or (args.logging_filter_expr is not None) or (args.logging_metadata_fields is not None):
        log_config = messages.SubnetworkLogConfig(enable=args.enable_flow_logs)
        if args.logging_aggregation_interval:
            log_config.aggregationInterval = flags.GetLoggingAggregationIntervalArg(messages).GetEnumForChoice(args.logging_aggregation_interval)
        if args.logging_flow_sampling is not None:
            log_config.flowSampling = args.logging_flow_sampling
        if args.logging_metadata:
            log_config.metadata = flags.GetLoggingMetadataArg(messages).GetEnumForChoice(args.logging_metadata)
        if args.logging_filter_expr is not None:
            log_config.filterExpr = args.logging_filter_expr
        if args.logging_metadata_fields is not None:
            log_config.metadataFields = args.logging_metadata_fields
        subnetwork.logConfig = log_config
    if include_alpha_logging:
        if args.enable_flow_logs is not None or args.aggregation_interval is not None or args.flow_sampling is not None or (args.metadata is not None):
            log_config = subnetwork.logConfig if subnetwork.logConfig is not None else messages.SubnetworkLogConfig(enable=args.enable_flow_logs)
            if args.aggregation_interval:
                log_config.aggregationInterval = flags.GetLoggingAggregationIntervalArgDeprecated(messages).GetEnumForChoice(args.aggregation_interval)
            if args.flow_sampling is not None:
                log_config.flowSampling = args.flow_sampling
            if args.metadata:
                log_config.metadata = flags.GetLoggingMetadataArgDeprecated(messages).GetEnumForChoice(args.metadata)
            if args.logging_filter_expr is not None:
                log_config.filterExpr = args.logging_filter_expr
            if args.logging_metadata_fields is not None:
                log_config.metadataFields = args.logging_metadata_fields
            subnetwork.logConfig = log_config
    if args.purpose:
        subnetwork.purpose = messages.Subnetwork.PurposeValueValuesEnum(args.purpose)
    if subnetwork.purpose == messages.Subnetwork.PurposeValueValuesEnum.INTERNAL_HTTPS_LOAD_BALANCER or subnetwork.purpose == messages.Subnetwork.PurposeValueValuesEnum.REGIONAL_MANAGED_PROXY or subnetwork.purpose == messages.Subnetwork.PurposeValueValuesEnum.GLOBAL_MANAGED_PROXY or (subnetwork.purpose == messages.Subnetwork.PurposeValueValuesEnum.PRIVATE_SERVICE_CONNECT) or (include_aggregate_purpose and subnetwork.purpose == messages.Subnetwork.PurposeValueValuesEnum.AGGREGATE):
        subnetwork.privateIpGoogleAccess = None
        subnetwork.enableFlowLogs = None
        subnetwork.logConfig = None
    if getattr(args, 'role', None):
        subnetwork.role = messages.Subnetwork.RoleValueValuesEnum(args.role)
    if args.private_ipv6_google_access_type is not None:
        subnetwork.privateIpv6GoogleAccess = flags.GetPrivateIpv6GoogleAccessTypeFlagMapper(messages).GetEnumForChoice(args.private_ipv6_google_access_type)
    if args.stack_type:
        subnetwork.stackType = messages.Subnetwork.StackTypeValueValuesEnum(args.stack_type)
    if args.ipv6_access_type:
        subnetwork.ipv6AccessType = messages.Subnetwork.Ipv6AccessTypeValueValuesEnum(args.ipv6_access_type)
    if include_l2 and args.enable_l2:
        subnetwork.enableL2 = True
        if args.vlan is not None:
            subnetwork.vlans.append(args.vlan)
    if args.reserved_internal_range:
        subnetwork.reservedInternalRange = args.reserved_internal_range
    if include_external_ipv6_prefix:
        if args.external_ipv6_prefix:
            subnetwork.externalIpv6Prefix = args.external_ipv6_prefix
    return subnetwork