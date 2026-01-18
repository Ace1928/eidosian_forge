from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListIpAddresses(base.ListCommand):
    """List internal IP addresses/ranges related a network."""
    example = '    List all internal IP addresses in a network:\n\n      $ {command} my-network\n\n    List IP addresses only for given types:\n\n      $ {command} my-network           --types=SUBNETWORK,PEER_USED,REMOTE_USED\n  '
    detailed_help = {'brief': 'List internal IP addresses in a network.', 'DESCRIPTION': '      *{command}* is used to list internal IP addresses in a network.\n      ', 'EXAMPLES': example}

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        parser.add_argument('name', help='The name of the network.')
        parser.add_argument('--types', type=lambda x: x.replace('-', '_').upper(), help='        Optional comma separate list of ip address types to filter on. Valid\n        values are `SUBNETWORK`, `RESERVED`, `PEER_USED`, `PEER_RESERVED`,\n        `REMOTE_USED` and `REMOTE_RESERVED`.\n        ')
        parser.display_info.AddFormat('        table(\n            type,\n            cidr:label=IP_RANGE,\n            region,\n            owner,\n            purpose)\n    ')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client.apitools_client
        messages = client.MESSAGES_MODULE
        project = properties.VALUES.core.project.Get(required=True)
        request = messages.ComputeNetworksListIpAddressesRequest(project=project, network=args.name, types=args.types)
        items = list_pager.YieldFromList(client.networks, request, method='ListIpAddresses', field='items', limit=args.limit, batch_size=None)
        return items