from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderFoldersLocationsRecommendersRecommendationsMarkSucceededRequest(_messages.Message):
    """A
  RecommenderFoldersLocationsRecommendersRecommendationsMarkSucceededRequest
  object.

  Fields:
    googleCloudRecommenderV1alpha2MarkRecommendationSucceededRequest: A
      GoogleCloudRecommenderV1alpha2MarkRecommendationSucceededRequest
      resource to be passed as the request body.
    name: Name of the recommendation.
  """
    googleCloudRecommenderV1alpha2MarkRecommendationSucceededRequest = _messages.MessageField('GoogleCloudRecommenderV1alpha2MarkRecommendationSucceededRequest', 1)
    name = _messages.StringField(2, required=True)