from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def SearchDataItems(self, request, global_params=None):
    """Searches DataItems in a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsSearchDataItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchDataItemsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchDataItems')
    return self._RunMethod(config, request, global_params=global_params)