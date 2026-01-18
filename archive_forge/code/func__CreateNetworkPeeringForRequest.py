from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute.networks.peerings import flags
from googlecloudsdk.core import properties
def _CreateNetworkPeeringForRequest(self, client, args):
    network_peering = client.messages.NetworkPeering(name=args.name, exportCustomRoutes=args.export_custom_routes, importCustomRoutes=args.import_custom_routes, exportSubnetRoutesWithPublicIp=args.export_subnet_routes_with_public_ip, importSubnetRoutesWithPublicIp=args.import_subnet_routes_with_public_ip)
    if getattr(args, 'stack_type'):
        network_peering.stackType = client.messages.NetworkPeering.StackTypeValueValuesEnum(args.stack_type)
    return network_peering