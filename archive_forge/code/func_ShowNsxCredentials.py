from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def ShowNsxCredentials(self, request, global_params=None):
    """Gets details of credentials for NSX appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsShowNsxCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Credentials) The response message.
      """
    config = self.GetMethodConfig('ShowNsxCredentials')
    return self._RunMethod(config, request, global_params=global_params)