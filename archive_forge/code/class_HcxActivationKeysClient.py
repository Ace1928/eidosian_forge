from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class HcxActivationKeysClient(util.VmwareClientBase):
    """cloud vmware hcx activation keys client."""

    def __init__(self):
        super(HcxActivationKeysClient, self).__init__()
        self.service = self.client.projects_locations_privateClouds_hcxActivationKeys

    def Create(self, hcx_activation_key):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysCreateRequest(parent=hcx_activation_key.Parent().RelativeName(), hcxActivationKeyId=hcx_activation_key.Name())
        return self.service.Create(request)

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetRequest(name=resource.RelativeName())
        return self.service.Get(request)

    def List(self, private_cloud_resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysListRequest(parent=private_cloud_resource.RelativeName())
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='hcxActivationKeys')