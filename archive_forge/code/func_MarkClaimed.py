from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
def MarkClaimed(self, request, global_params=None):
    """Marks the Recommendation State as Claimed. Users can use this method to indicate to the Recommender API that they are starting to apply the recommendation themselves. This stops the recommendation content from being updated. Associated insights are frozen and placed in the ACCEPTED state. MarkRecommendationClaimed can be applied to recommendations in CLAIMED or ACTIVE state. Requires the recommender.*.update IAM permission for the specified recommender.

      Args:
        request: (RecommenderProjectsLocationsRecommendersRecommendationsMarkClaimedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecommenderV1alpha2Recommendation) The response message.
      """
    config = self.GetMethodConfig('MarkClaimed')
    return self._RunMethod(config, request, global_params=global_params)