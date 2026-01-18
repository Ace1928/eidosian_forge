from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags as routers_flags
class GetNatMappingInfo(base.ListCommand):
    """Display NAT Mapping information in a router."""
    ROUTER_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.ROUTER_ARG = routers_flags.RouterArgument()
        cls.ROUTER_ARG.AddArgument(parser, operation_type='get NAT mapping info')
        routers_flags.AddGetNatMappingInfoArgs(parser)
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        router_ref = self.ROUTER_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        params = router_ref.AsDict()
        if args.nat_name:
            params['natName'] = args.nat_name
        request = client.messages.ComputeRoutersGetNatMappingInfoRequest(**params)
        return list_pager.YieldFromList(client.apitools_client.routers, request, limit=args.limit, batch_size=args.page_size, method='GetNatMappingInfo', field='result', current_token_attribute='pageToken', next_token_attribute='nextPageToken', batch_size_attribute='maxResults')