from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GetNetworkInterfaces(self, args, client, holder):
    if args.network_interface:
        return instance_template_utils.CreateNetworkInterfaceMessages(resources=holder.resources, scope_lister=flags.GetDefaultScopeLister(client), messages=client.messages, network_interface_arg=args.network_interface, subnet_region=args.region)
    stack_type = getattr(args, 'stack_type', None)
    ipv6_network_tier = getattr(args, 'ipv6_network_tier', None)
    ipv6_address = getattr(args, 'ipv6_address', None)
    ipv6_prefix_length = getattr(args, 'ipv6_prefix_length', None)
    external_ipv6_address = getattr(args, 'external_ipv6_address', None)
    external_ipv6_prefix_length = getattr(args, 'external_ipv6_prefix_length', None)
    internal_ipv6_address = getattr(args, 'internal_ipv6_address', None)
    internal_ipv6_prefix_length = getattr(args, 'internal_ipv6_prefix_length', None)
    return [instance_template_utils.CreateNetworkInterfaceMessage(resources=holder.resources, scope_lister=flags.GetDefaultScopeLister(client), messages=client.messages, network=args.network, private_ip=args.private_network_ip, subnet_region=args.region, subnet=args.subnet, address=instance_template_utils.EPHEMERAL_ADDRESS if not args.no_address and (not args.address) else args.address, network_tier=getattr(args, 'network_tier', None), stack_type=stack_type, ipv6_network_tier=ipv6_network_tier, ipv6_address=ipv6_address, ipv6_prefix_length=ipv6_prefix_length, external_ipv6_address=external_ipv6_address, external_ipv6_prefix_length=external_ipv6_prefix_length, internal_ipv6_address=internal_ipv6_address, internal_ipv6_prefix_length=internal_ipv6_prefix_length)]