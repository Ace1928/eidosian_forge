from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderFoldersLocationsRecommendersRecommendationsMarkFailedRequest(_messages.Message):
    """A
  RecommenderFoldersLocationsRecommendersRecommendationsMarkFailedRequest
  object.

  Fields:
    googleCloudRecommenderV1alpha2MarkRecommendationFailedRequest: A
      GoogleCloudRecommenderV1alpha2MarkRecommendationFailedRequest resource
      to be passed as the request body.
    name: Name of the recommendation.
  """
    googleCloudRecommenderV1alpha2MarkRecommendationFailedRequest = _messages.MessageField('GoogleCloudRecommenderV1alpha2MarkRecommendationFailedRequest', 1)
    name = _messages.StringField(2, required=True)