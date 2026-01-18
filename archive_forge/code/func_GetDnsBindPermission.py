from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def GetDnsBindPermission(self, request, global_params=None):
    """Gets all the principals having bind permission on the intranet VPC associated with the consumer project granted by the Grant API. DnsBindPermission is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsGetDnsBindPermissionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsBindPermission) The response message.
      """
    config = self.GetMethodConfig('GetDnsBindPermission')
    return self._RunMethod(config, request, global_params=global_params)