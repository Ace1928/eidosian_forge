from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ScoreComponentRecommendation(_messages.Message):
    """Recommendation based on security concerns and score.

  Fields:
    actions: Actions for the recommendation to improve the security score.
    description: Description of the recommendation.
    impact: Potential impact of this recommendation on the overall score. This
      denotes how important this recommendation is to improve the score.
    title: Title represents recommendation title.
  """
    actions = _messages.MessageField('GoogleCloudApigeeV1ScoreComponentRecommendationAction', 1, repeated=True)
    description = _messages.StringField(2)
    impact = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    title = _messages.StringField(4)