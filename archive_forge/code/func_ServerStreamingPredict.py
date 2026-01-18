from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ServerStreamingPredict(self, request, global_params=None):
    """Perform a server-side streaming online prediction request for Vertex LLM streaming.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsServerStreamingPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1StreamingPredictResponse) The response message.
      """
    config = self.GetMethodConfig('ServerStreamingPredict')
    return self._RunMethod(config, request, global_params=global_params)