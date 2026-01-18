from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class NetworkPeeringRoutesClient(util.VmwareClientBase):
    """VMware Engine VPC network peering routes client."""

    def __init__(self):
        super(NetworkPeeringRoutesClient, self).__init__()
        self.service = self.client.projects_locations_networkPeerings_peeringRoutes

    def List(self, network_peering):
        networkpeering = network_peering.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsNetworkPeeringsPeeringRoutesListRequest(parent=networkpeering)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='peeringRoutes')