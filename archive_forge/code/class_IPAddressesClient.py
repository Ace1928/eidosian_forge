from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware.sddc import util
from googlecloudsdk.command_lib.vmware.sddc import flags
import six.moves.urllib.parse
class IPAddressesClient(util.VmwareClientBase):
    """cloud vmware sddc ip addresses client."""

    def __init__(self):
        super(IPAddressesClient, self).__init__()
        self.service = self.client.projects_locations_clusterGroups_ipAddresses

    def Create(self, resource, internal_ip, labels=None):
        ip_address = self.messages.IpAddress(internalIp=internal_ip)
        flags.AddLabelsToMessage(labels, ip_address)
        request = self.messages.SddcProjectsLocationsClusterGroupsIpAddressesCreateRequest(ipAddress=ip_address, ipAddressId=resource.Name(), parent=self.GetResourcePath(resource, resource_path=resource.Parent().RelativeName()))
        return self.service.Create(request)

    def Delete(self, resource):
        request = self.messages.SddcProjectsLocationsClusterGroupsIpAddressesDeleteRequest(name=self.GetResourcePath(resource, resource_path=resource.RelativeName()))
        return self.service.Delete(request)

    def Get(self, resource):
        request = self.messages.SddcProjectsLocationsClusterGroupsIpAddressesGetRequest(name=self.GetResourcePath(resource, resource_path=resource.RelativeName()))
        return self.service.Get(request)

    def GetResourcePath(self, resource, resource_path, encoded_cluster_groups_id=False):
        result = six.text_type(resource_path)
        if '/' not in resource.clusterGroupsId:
            return result
        cluster_groups_id = resource.clusterGroupsId.split('/').pop()
        cluster_groups_id_path = six.text_type(resource.clusterGroupsId)
        if encoded_cluster_groups_id:
            cluster_groups_id_path = six.moves.urllib.parse.quote(cluster_groups_id_path, safe='')
        return result.replace(cluster_groups_id_path, cluster_groups_id)

    def List(self, resource):
        ip_name = resource.RelativeName()
        request = self.messages.SddcProjectsLocationsClusterGroupsIpAddressesListRequest(parent=ip_name)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='ipAddresses')