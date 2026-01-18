from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.api_lib.vmware.networks import NetworksClient
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetServiceNetwork(self, type_enum, service_network=None):
    if service_network:
        return service_network
    if type_enum == self.messages.PrivateConnection.TypeValueValuesEnum.PRIVATE_SERVICE_ACCESS:
        return 'servicenetworking'
    if type_enum == self.messages.PrivateConnection.TypeValueValuesEnum.DELL_POWERSCALE:
        return 'dell-tenant-vpc'
    if type_enum == self.messages.PrivateConnection.TypeValueValuesEnum.NETAPP_CLOUD_VOLUMES:
        return 'netapp-tenant-vpc'