from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class PrivateConnectionPeeringRoutesClient(util.VmwareClientBase):
    """Google Cloud private connections peering routes client."""

    def __init__(self):
        super(PrivateConnectionPeeringRoutesClient, self).__init__()
        self.service = self.client.projects_locations_privateConnections_peeringRoutes

    def List(self, private_connection):
        private_connection = private_connection.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsPrivateConnectionsPeeringRoutesListRequest(parent=private_connection)
        return list_pager.YieldFromList(self.service, request, field='peeringRoutes', batch_size_attribute='pageSize')