from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def RemoveGitLabConnectedRepository(self, request, global_params=None):
    """Remove a GitLab repository from a given GitLabConfig's connected repositories. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsRemoveGitLabConnectedRepositoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('RemoveGitLabConnectedRepository')
    return self._RunMethod(config, request, global_params=global_params)