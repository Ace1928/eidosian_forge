from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
def FetchLinkableRepositories(self, request, global_params=None):
    """FetchLinkableRepositories get repositories from SCM that are accessible and could be added to the connection.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsFetchLinkableRepositoriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchLinkableRepositoriesResponse) The response message.
      """
    config = self.GetMethodConfig('FetchLinkableRepositories')
    return self._RunMethod(config, request, global_params=global_params)