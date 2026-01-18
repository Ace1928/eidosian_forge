from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class NodesClient(util.VmwareClientBase):
    """cloud vmware cluster nodes client."""

    def __init__(self):
        super(NodesClient, self).__init__()
        self.service = self.client.projects_locations_privateClouds_clusters_nodes

    def List(self, cluster_resource):
        cluster = cluster_resource.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsClustersNodesListRequest(parent=cluster)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='nodes')

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsClustersNodesGetRequest(name=resource.RelativeName())
        return self.service.Get(request)