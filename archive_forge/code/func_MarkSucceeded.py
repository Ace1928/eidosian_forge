from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
def MarkSucceeded(self, request, global_params=None):
    """Marks the Recommendation State as Succeeded. Users can use this method to indicate to the Recommender API that they have applied the recommendation themselves, and the operation was successful. This stops the recommendation content from being updated. Associated insights are frozen and placed in the ACCEPTED state. MarkRecommendationSucceeded can be applied to recommendations in ACTIVE, CLAIMED, SUCCEEDED, or FAILED state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderProjectsLocationsRecommendersRecommendationsMarkSucceededRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
    config = self.GetMethodConfig('MarkSucceeded')
    return self._RunMethod(config, request, global_params=global_params)