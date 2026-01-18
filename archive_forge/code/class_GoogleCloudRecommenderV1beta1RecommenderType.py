from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1RecommenderType(_messages.Message):
    """The type of a recommender.

  Fields:
    name: The recommender's name in format RecommenderTypes/{recommender_type}
      eg: recommenderTypes/google.iam.policy.Recommender
  """
    name = _messages.StringField(1)