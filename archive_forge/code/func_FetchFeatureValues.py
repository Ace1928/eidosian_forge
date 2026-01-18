from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def FetchFeatureValues(self, request, global_params=None):
    """Fetch feature values under a FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFetchFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1FetchFeatureValuesResponse) The response message.
      """
    config = self.GetMethodConfig('FetchFeatureValues')
    return self._RunMethod(config, request, global_params=global_params)