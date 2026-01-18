from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def ShowVcenterCredentials(self, request, global_params=None):
    """Gets details of credentials for Vcenter appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsShowVcenterCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Credentials) The response message.
      """
    config = self.GetMethodConfig('ShowVcenterCredentials')
    return self._RunMethod(config, request, global_params=global_params)