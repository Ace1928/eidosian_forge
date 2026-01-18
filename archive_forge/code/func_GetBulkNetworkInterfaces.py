from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def GetBulkNetworkInterfaces(args, resource_parser, compute_client, holder, project, location, scope, skip_defaults):
    """Gets network interfaces in bulk instance API."""
    bulk_args = ['network_interface', 'network', 'network_tier', 'subnet', 'no_address', 'stack_type']
    if skip_defaults and (not instance_utils.IsAnySpecified(args, *bulk_args)):
        return []
    elif args.network_interface:
        return CreateNetworkInterfaceMessages(resources=resource_parser, compute_client=compute_client, network_interface_arg=args.network_interface, project=project, location=location, scope=scope)
    else:
        return [instances_utils.CreateNetworkInterfaceMessage(resources=holder.resources, compute_client=compute_client, network=args.network, subnet=args.subnet, no_address=args.no_address, project=project, location=location, scope=scope, network_tier=getattr(args, 'network_tier', None), stack_type=getattr(args, 'stack_type', None))]