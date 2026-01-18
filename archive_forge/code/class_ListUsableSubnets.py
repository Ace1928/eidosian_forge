from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListUsableSubnets(base.ListCommand):
    """List subnetworks which the current user has permission to use."""
    enable_service_project = False

    @staticmethod
    def _EnableComputeApi():
        return properties.VALUES.compute.use_new_list_usable_subnets_api.GetBool()

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddFormat('        table(\n          subnetwork.segment(-5):label=PROJECT,\n          subnetwork.segment(-3):label=REGION,\n          network.segment(-1):label=NETWORK,\n          subnetwork.segment(-1):label=SUBNET,\n          ipCidrRange:label=RANGE,\n          secondaryIpRanges.map().format("{0} {1}", rangeName, ipCidrRange).list(separator="\n"):label=SECONDARY_RANGES,\n          purpose,\n          role,\n          stackType,\n          ipv6AccessType,\n          internalIpv6Prefix,\n          externalIpv6Prefix\n        )')
        if cls.enable_service_project:
            parser.add_argument('--service-project', required=False, help='          The project id or project number in which the subnetwork is intended to be\n          used. Only applied for Shared VPC.\n          See [Shared VPC documentation](https://cloud.google.com/vpc/docs/shared-vpc/).\n          ')

    def Collection(self):
        return 'compute.subnetworks'

    def GetUriFunc(self):

        def _GetUri(search_result):
            return ''.join([p.value.string_value for p in search_result.resource.additionalProperties if p.key == 'selfLink'])
        return _GetUri

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        messages = holder.client.messages
        request = messages.ComputeSubnetworksListUsableRequest(project=properties.VALUES.core.project.Get(required=True))
        if self.enable_service_project and args.service_project:
            request.serviceProject = args.service_project
        return list_pager.YieldFromList(client.apitools_client.subnetworks, request, method='ListUsable', batch_size_attribute='maxResults', batch_size=500, field='items')