from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gsuiteaddons.v1 import gsuiteaddons_v1_messages as messages
def GetInstallStatus(self, request, global_params=None):
    """Gets the install status of a test deployment.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsGetInstallStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1InstallStatus) The response message.
      """
    config = self.GetMethodConfig('GetInstallStatus')
    return self._RunMethod(config, request, global_params=global_params)