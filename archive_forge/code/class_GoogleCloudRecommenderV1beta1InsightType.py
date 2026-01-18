from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1InsightType(_messages.Message):
    """The type of insight.

  Fields:
    name: The insight_type's name in format insightTypes/{insight_type} eg:
      insightTypes/google.iam.policy.Insight
  """
    name = _messages.StringField(1)