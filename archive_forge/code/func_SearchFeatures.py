from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def SearchFeatures(self, request, global_params=None):
    """Searches Features matching a query in a given project.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresSearchFeaturesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchFeaturesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchFeatures')
    return self._RunMethod(config, request, global_params=global_params)