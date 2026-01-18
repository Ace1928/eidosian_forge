from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def GetApp(self, request, global_params=None):
    """Get the GitHub App associated with a GitHub Enterprise Config. Uses the GitHub API: https://developer.github.com/enterprise/2.21/v3/apps/#get-an-app This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsGetAppRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GitHubEnterpriseApp) The response message.
      """
    config = self.GetMethodConfig('GetApp')
    return self._RunMethod(config, request, global_params=global_params)