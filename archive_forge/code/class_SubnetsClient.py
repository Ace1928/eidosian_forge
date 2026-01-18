from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class SubnetsClient(util.VmwareClientBase):
    """cloud vmware private-clouds subnets client."""

    def __init__(self):
        super(SubnetsClient, self).__init__()
        self.service = self.client.projects_locations_privateClouds_subnets

    def List(self, resource):
        address_name = resource.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsSubnetsListRequest(parent=address_name)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='subnets')

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsSubnetsGetRequest(name=resource.RelativeName())
        response = self.service.Get(request)
        return response

    def Update(self, resource, ip_cidr_range):
        subnet = self.Get(resource)
        subnet.ipCidrRange = ip_cidr_range
        update_mask = ['ip_cidr_range']
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsSubnetsPatchRequest(subnet=subnet, name=resource.RelativeName(), updateMask=','.join(update_mask))
        return self.service.Patch(request)