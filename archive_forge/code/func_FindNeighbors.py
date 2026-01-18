from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def FindNeighbors(self, request, global_params=None):
    """Finds the nearest neighbors of each vector within the request.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsFindNeighborsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1FindNeighborsResponse) The response message.
      """
    config = self.GetMethodConfig('FindNeighbors')
    return self._RunMethod(config, request, global_params=global_params)