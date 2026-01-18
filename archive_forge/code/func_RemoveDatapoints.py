from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def RemoveDatapoints(self, request, global_params=None):
    """Remove Datapoints from an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesRemoveDatapointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1RemoveDatapointsResponse) The response message.
      """
    config = self.GetMethodConfig('RemoveDatapoints')
    return self._RunMethod(config, request, global_params=global_params)