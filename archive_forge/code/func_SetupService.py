from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privilegedaccessmanager.v1alpha import privilegedaccessmanager_v1alpha_messages as messages
def SetupService(self, request, global_params=None):
    """SetupService sets up PAM service for a GCP project/folder/organization. This needs to be done before Entitlements parented under the project/folder/organization can be created.

      Args:
        request: (PrivilegedaccessmanagerProjectsLocationsSetupServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SetupServiceResponse) The response message.
      """
    config = self.GetMethodConfig('SetupService')
    return self._RunMethod(config, request, global_params=global_params)