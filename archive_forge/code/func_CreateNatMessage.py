from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def CreateNatMessage(args, compute_holder):
    """Creates a NAT message from the specified arguments."""
    params = {'name': args.name}
    params['sourceSubnetworkIpRangesToNat'], params['subnetworks'] = _ParseSubnetFields(args, compute_holder)
    if args.type is not None:
        params['type'] = compute_holder.client.messages.RouterNat.TypeValueValuesEnum(args.type)
    is_private = args.type == 'PRIVATE'
    is_ip_allocation_specified = args.auto_allocate_nat_external_ips or args.nat_external_ip_pool
    if is_private:
        if is_ip_allocation_specified:
            raise IpAllocateOptionShouldNotBeSpecifiedError()
    else:
        if not is_ip_allocation_specified:
            raise IpAllocationUnspecifiedError()
        option, nat_ips = _ParseNatIpFields(args, compute_holder)
        params['natIpAllocateOption'] = option
        params['natIps'] = nat_ips
    if args.auto_network_tier is not None:
        params['autoNetworkTier'] = compute_holder.client.messages.RouterNat.AutoNetworkTierValueValuesEnum(args.auto_network_tier)
    if args.endpoint_types is not None:
        params['endpointTypes'] = [compute_holder.client.messages.RouterNat.EndpointTypesValueListEntryValuesEnum(endpoint_type) for endpoint_type in args.endpoint_types]
    params['udpIdleTimeoutSec'] = args.udp_idle_timeout
    params['icmpIdleTimeoutSec'] = args.icmp_idle_timeout
    params['tcpEstablishedIdleTimeoutSec'] = args.tcp_established_idle_timeout
    params['tcpTransitoryIdleTimeoutSec'] = args.tcp_transitory_idle_timeout
    params['tcpTimeWaitTimeoutSec'] = args.tcp_time_wait_timeout
    params['minPortsPerVm'] = args.min_ports_per_vm
    params['maxPortsPerVm'] = args.max_ports_per_vm
    params['enableDynamicPortAllocation'] = args.enable_dynamic_port_allocation
    if args.enable_logging is not None or args.log_filter is not None:
        log_config = compute_holder.client.messages.RouterNatLogConfig()
        log_config.enable = args.enable_logging
        if args.log_filter is not None:
            log_config.filter = _TranslateLogFilter(args.log_filter, compute_holder)
        params['logConfig'] = log_config
    if args.enable_endpoint_independent_mapping is not None:
        params['enableEndpointIndependentMapping'] = args.enable_endpoint_independent_mapping
    if args.rules:
        params['rules'] = _ParseRulesFromYamlFile(args.rules, compute_holder)
    return compute_holder.client.messages.RouterNat(**params)