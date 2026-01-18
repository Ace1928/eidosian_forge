from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
def AccessReadToken(self, request, global_params=None):
    """Fetches read token of a given repository.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsRepositoriesAccessReadTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchReadTokenResponse) The response message.
      """
    config = self.GetMethodConfig('AccessReadToken')
    return self._RunMethod(config, request, global_params=global_params)