from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def StreamRawPredict(self, request, global_params=None):
    """Perform a streaming online prediction with an arbitrary HTTP payload.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsStreamRawPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
    config = self.GetMethodConfig('StreamRawPredict')
    return self._RunMethod(config, request, global_params=global_params)