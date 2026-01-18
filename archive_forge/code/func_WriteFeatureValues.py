from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def WriteFeatureValues(self, request, global_params=None):
    """Writes Feature values of one or more entities of an EntityType. The Feature values are merged into existing entities if any. The Feature values to be written must have timestamp within the online storage retention.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesWriteFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1WriteFeatureValuesResponse) The response message.
      """
    config = self.GetMethodConfig('WriteFeatureValues')
    return self._RunMethod(config, request, global_params=global_params)