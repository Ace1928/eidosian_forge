from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def StreamingReadFeatureValues(self, request, global_params=None):
    """Reads Feature values for multiple entities. Depending on their size, data for different entities may be broken up across multiple responses.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesStreamingReadFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadFeatureValuesResponse) The response message.
      """
    config = self.GetMethodConfig('StreamingReadFeatureValues')
    return self._RunMethod(config, request, global_params=global_params)