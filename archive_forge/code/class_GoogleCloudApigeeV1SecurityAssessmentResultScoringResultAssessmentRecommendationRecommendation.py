from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAssessmentRecommendationRecommendation(_messages.Message):
    """The format of the assessment recommendation.

  Fields:
    description: The description of the recommendation.
    link: The link for the recommendation.
  """
    description = _messages.StringField(1)
    link = _messages.MessageField('GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAssessmentRecommendationRecommendationLink', 2)