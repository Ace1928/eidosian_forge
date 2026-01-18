from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def Predict(self, request, global_params=None):
    """Perform an online prediction.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1PredictResponse) The response message.
      """
    config = self.GetMethodConfig('Predict')
    return self._RunMethod(config, request, global_params=global_params)