from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class ExternalAddressesClient(util.VmwareClientBase):
    """cloud vmware external addresses client."""

    def __init__(self):
        super(ExternalAddressesClient, self).__init__()
        self.service = self.client.projects_locations_privateClouds_externalAddresses

    def Create(self, resource, internal_ip, description=None):
        external_address = self.messages.ExternalAddress(internalIp=internal_ip, description=description)
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsExternalAddressesCreateRequest(externalAddress=external_address, externalAddressId=resource.Name(), parent=resource.Parent().RelativeName())
        return self.service.Create(request)

    def Update(self, resource, internal_ip=None, description=None):
        external_address = self.Get(resource)
        update_mask = []
        if description is not None:
            external_address.description = description
            update_mask.append('description')
        if internal_ip is not None:
            external_address.internalIp = internal_ip
            update_mask.append('internal_ip')
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsExternalAddressesPatchRequest(externalAddress=external_address, name=resource.RelativeName(), updateMask=','.join(update_mask))
        return self.service.Patch(request)

    def Delete(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsExternalAddressesDeleteRequest(name=resource.RelativeName())
        return self.service.Delete(request)

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsExternalAddressesGetRequest(name=resource.RelativeName())
        return self.service.Get(request)

    def List(self, resource):
        address_name = resource.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsExternalAddressesListRequest(parent=address_name)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='externalAddresses')