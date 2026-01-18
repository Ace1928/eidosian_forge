from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def SearchNearestEntities(self, request, global_params=None):
    """Search the nearest entities under a FeatureView. Search only works for indexable feature view; if a feature view isn't indexable, returns Invalid argument response.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSearchNearestEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchNearestEntitiesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchNearestEntities')
    return self._RunMethod(config, request, global_params=global_params)