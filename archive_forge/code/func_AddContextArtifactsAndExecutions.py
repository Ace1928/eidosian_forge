from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def AddContextArtifactsAndExecutions(self, request, global_params=None):
    """Adds a set of Artifacts and Executions to a Context. If any of the Artifacts or Executions have already been added to a Context, they are simply skipped.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsAddContextArtifactsAndExecutionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AddContextArtifactsAndExecutionsResponse) The response message.
      """
    config = self.GetMethodConfig('AddContextArtifactsAndExecutions')
    return self._RunMethod(config, request, global_params=global_params)