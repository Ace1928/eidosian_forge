from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def Sync(self, request, global_params=None):
    """Triggers on-demand sync for the FeatureView.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSyncRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1SyncFeatureViewResponse) The response message.
      """
    config = self.GetMethodConfig('Sync')
    return self._RunMethod(config, request, global_params=global_params)