from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def ResetNsxCredentials(self, request, global_params=None):
    """Resets credentials of the NSX appliance.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsResetNsxCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ResetNsxCredentials')
    return self._RunMethod(config, request, global_params=global_params)