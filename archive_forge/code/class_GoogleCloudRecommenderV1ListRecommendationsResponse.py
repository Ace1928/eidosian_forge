from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1ListRecommendationsResponse(_messages.Message):
    """Response to the `ListRecommendations` method.

  Fields:
    nextPageToken: A token that can be used to request the next page of
      results. This field is empty if there are no additional results.
    recommendations: The set of recommendations for the `parent` resource.
  """
    nextPageToken = _messages.StringField(1)
    recommendations = _messages.MessageField('GoogleCloudRecommenderV1Recommendation', 2, repeated=True)