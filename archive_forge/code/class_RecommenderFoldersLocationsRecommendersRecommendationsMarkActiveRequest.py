from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderFoldersLocationsRecommendersRecommendationsMarkActiveRequest(_messages.Message):
    """A
  RecommenderFoldersLocationsRecommendersRecommendationsMarkActiveRequest
  object.

  Fields:
    googleCloudRecommenderV1alpha2MarkRecommendationActiveRequest: A
      GoogleCloudRecommenderV1alpha2MarkRecommendationActiveRequest resource
      to be passed as the request body.
    name: Name of the recommendation.
  """
    googleCloudRecommenderV1alpha2MarkRecommendationActiveRequest = _messages.MessageField('GoogleCloudRecommenderV1alpha2MarkRecommendationActiveRequest', 1)
    name = _messages.StringField(2, required=True)