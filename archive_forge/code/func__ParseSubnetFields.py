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
def _ParseSubnetFields(args, compute_holder):
    """Parses arguments related to subnets to use for NAT."""
    subnetworks = list()
    messages = compute_holder.client.messages
    if args.subnet_option == nat_flags.SubnetOption.ALL_RANGES:
        ranges_to_nat = messages.RouterNat.SourceSubnetworkIpRangesToNatValueValuesEnum.ALL_SUBNETWORKS_ALL_IP_RANGES
    elif args.subnet_option == nat_flags.SubnetOption.PRIMARY_RANGES:
        ranges_to_nat = messages.RouterNat.SourceSubnetworkIpRangesToNatValueValuesEnum.ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES
    else:
        ranges_to_nat = messages.RouterNat.SourceSubnetworkIpRangesToNatValueValuesEnum.LIST_OF_SUBNETWORKS
        subnet_usages = dict()
        for custom_subnet_arg in args.nat_custom_subnet_ip_ranges:
            colons = custom_subnet_arg.count(':')
            range_option = None
            if colons > 1:
                raise calliope_exceptions.InvalidArgumentException('--nat-custom-subnet-ip-ranges', 'Each specified subnet must be of the form SUBNETWORK or SUBNETWORK:RANGE_NAME')
            elif colons == 1:
                subnet_name, range_option = custom_subnet_arg.split(':')
            else:
                subnet_name = custom_subnet_arg
            if subnet_name not in subnet_usages:
                subnet_usages[subnet_name] = SubnetUsage()
            if range_option is not None:
                if range_option == 'ALL':
                    subnet_usages[subnet_name].using_all = True
                else:
                    subnet_usages[subnet_name].secondary_ranges.append(range_option)
            else:
                subnet_usages[subnet_name].using_primary = True
        for subnet_name in subnet_usages:
            subnet_ref = subnet_flags.SubnetworkResolver().ResolveResources([subnet_name], compute_scope.ScopeEnum.REGION, args.region, compute_holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(compute_holder.client))
            subnet_usage = subnet_usages[subnet_name]
            options = []
            if subnet_usage.using_all:
                options.append(messages.RouterNatSubnetworkToNat.SourceIpRangesToNatValueListEntryValuesEnum.ALL_IP_RANGES)
            if subnet_usage.using_primary:
                options.append(messages.RouterNatSubnetworkToNat.SourceIpRangesToNatValueListEntryValuesEnum.PRIMARY_IP_RANGE)
            if subnet_usage.secondary_ranges:
                options.append(messages.RouterNatSubnetworkToNat.SourceIpRangesToNatValueListEntryValuesEnum.LIST_OF_SECONDARY_IP_RANGES)
            subnetworks.append({'name': six.text_type(subnet_ref[0]), 'sourceIpRangesToNat': options, 'secondaryIpRangeNames': subnet_usage.secondary_ranges})
    return (ranges_to_nat, sorted(subnetworks, key=lambda subnet: subnet['name']))