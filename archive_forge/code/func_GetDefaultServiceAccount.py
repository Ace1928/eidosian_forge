from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def GetDefaultServiceAccount(self, request, global_params=None):
    """Returns the `DefaultServiceAccount` used by the project.

      Args:
        request: (CloudbuildProjectsLocationsGetDefaultServiceAccountRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DefaultServiceAccount) The response message.
      """
    config = self.GetMethodConfig('GetDefaultServiceAccount')
    return self._RunMethod(config, request, global_params=global_params)