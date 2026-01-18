from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AttributionScoreSkewThresholdsValue(_messages.Message):
    """Key is the feature name and value is the threshold. The threshold here
    is against attribution score distance between the training and prediction
    feature.

    Messages:
      AdditionalProperty: An additional property for a
        AttributionScoreSkewThresholdsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AttributionScoreSkewThresholdsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AttributionScoreSkewThresholdsValue
      object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1ThresholdConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1ThresholdConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)