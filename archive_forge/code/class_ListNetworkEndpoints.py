from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.network_endpoint_groups import flags
from googlecloudsdk.core.resource import resource_projection_spec
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListNetworkEndpoints(base.ListCommand):
    """List network endpoints in a network endpoint group."""
    detailed_help = DETAILED_HELP
    display_info_format = '        table(\n          networkEndpoint.instance,\n          networkEndpoint.ipAddress,\n          networkEndpoint.port,\n          networkEndpoint.fqdn\n        )'

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddFormat(cls.display_info_format)
        base.URI_FLAG.RemoveFromParser(parser)
        flags.MakeNetworkEndpointGroupsArg().AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        messages = client.messages
        neg_ref = flags.MakeNetworkEndpointGroupsArg().ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        display_info = args.GetDisplayInfo()
        defaults = resource_projection_spec.ProjectionSpec(symbols=display_info.transforms, aliases=display_info.aliases)
        args.filter, filter_expr = filter_rewrite.Rewriter().Rewrite(args.filter, defaults=defaults)
        if hasattr(neg_ref, 'zone'):
            request = messages.ComputeNetworkEndpointGroupsListNetworkEndpointsRequest(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, zone=neg_ref.zone, filter=filter_expr)
            service = client.apitools_client.networkEndpointGroups
        elif hasattr(neg_ref, 'region'):
            request = messages.ComputeRegionNetworkEndpointGroupsListNetworkEndpointsRequest(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, region=neg_ref.region, filter=filter_expr)
            service = client.apitools_client.regionNetworkEndpointGroups
        else:
            request = messages.ComputeGlobalNetworkEndpointGroupsListNetworkEndpointsRequest(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, filter=filter_expr)
            service = client.apitools_client.globalNetworkEndpointGroups
        return list_pager.YieldFromList(service, request, method='ListNetworkEndpoints', field='items', limit=args.limit, batch_size=None)