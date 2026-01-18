from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def StreamGenerateContent(self, request, global_params=None):
    """Generate content with multimodal inputs with streaming support.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsStreamGenerateContentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1GenerateContentResponse) The response message.
      """
    config = self.GetMethodConfig('StreamGenerateContent')
    return self._RunMethod(config, request, global_params=global_params)