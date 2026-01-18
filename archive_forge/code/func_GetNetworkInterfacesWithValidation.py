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
def GetNetworkInterfacesWithValidation(args, resource_parser, compute_client, holder, project, location, scope, skip_defaults, support_public_dns=False, support_ipv6_assignment=False, support_internal_ipv6_reservation=False):
    """Validates and retrieves the network interface message."""
    network_interface_from_file = getattr(args, 'network_interface_from_file', None)
    network_interface_from_json_string = getattr(args, 'network_interface_from_json_string', None)
    if args.network_interface or network_interface_from_file or network_interface_from_json_string:
        return CreateNetworkInterfaceMessages(resources=resource_parser, compute_client=compute_client, network_interface_arg=args.network_interface, network_interface_json=network_interface_from_file if network_interface_from_file is not None else network_interface_from_json_string, project=project, location=location, scope=scope, support_internal_ipv6_reservation=support_internal_ipv6_reservation)
    else:
        instances_flags.ValidatePublicPtrFlags(args)
        if support_public_dns or support_ipv6_assignment:
            if support_public_dns:
                instances_flags.ValidatePublicDnsFlags(args)
            return GetNetworkInterfacesAlpha(args, compute_client, holder, project, location, scope, skip_defaults)
        return GetNetworkInterfaces(args, compute_client, holder, project, location, scope, skip_defaults, support_internal_ipv6_reservation=support_internal_ipv6_reservation)