from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gsuiteaddons.v1 import gsuiteaddons_v1_messages as messages
def GetAuthorization(self, request, global_params=None):
    """Gets the authorization information for deployments in a given project.

      Args:
        request: (GsuiteaddonsProjectsGetAuthorizationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1Authorization) The response message.
      """
    config = self.GetMethodConfig('GetAuthorization')
    return self._RunMethod(config, request, global_params=global_params)