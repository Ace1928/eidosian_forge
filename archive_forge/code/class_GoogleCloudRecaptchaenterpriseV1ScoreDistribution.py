from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1ScoreDistribution(_messages.Message):
    """Score distribution.

  Messages:
    ScoreBucketsValue: Map key is score value multiplied by 100. The scores
      are discrete values between [0, 1]. The maximum number of buckets is on
      order of a few dozen, but typically much lower (ie. 10).

  Fields:
    scoreBuckets: Map key is score value multiplied by 100. The scores are
      discrete values between [0, 1]. The maximum number of buckets is on
      order of a few dozen, but typically much lower (ie. 10).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ScoreBucketsValue(_messages.Message):
        """Map key is score value multiplied by 100. The scores are discrete
    values between [0, 1]. The maximum number of buckets is on order of a few
    dozen, but typically much lower (ie. 10).

    Messages:
      AdditionalProperty: An additional property for a ScoreBucketsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ScoreBucketsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ScoreBucketsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    scoreBuckets = _messages.MessageField('ScoreBucketsValue', 1)