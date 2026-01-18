from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelMonitoringObjectiveConfigTrainingPredictionSkewDetectionConfig(_messages.Message):
    """The config for Training & Prediction data skew detection. It specifies
  the training dataset sources and the skew detection parameters.

  Messages:
    AttributionScoreSkewThresholdsValue: Key is the feature name and value is
      the threshold. The threshold here is against attribution score distance
      between the training and prediction feature.
    SkewThresholdsValue: Key is the feature name and value is the threshold.
      If a feature needs to be monitored for skew, a value threshold must be
      configured for that feature. The threshold here is against feature
      distribution distance between the training and prediction feature.

  Fields:
    attributionScoreSkewThresholds: Key is the feature name and value is the
      threshold. The threshold here is against attribution score distance
      between the training and prediction feature.
    defaultSkewThreshold: Skew anomaly detection threshold used by all
      features. When the per-feature thresholds are not set, this field can be
      used to specify a threshold for all features.
    skewThresholds: Key is the feature name and value is the threshold. If a
      feature needs to be monitored for skew, a value threshold must be
      configured for that feature. The threshold here is against feature
      distribution distance between the training and prediction feature.
  """

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SkewThresholdsValue(_messages.Message):
        """Key is the feature name and value is the threshold. If a feature needs
    to be monitored for skew, a value threshold must be configured for that
    feature. The threshold here is against feature distribution distance
    between the training and prediction feature.

    Messages:
      AdditionalProperty: An additional property for a SkewThresholdsValue
        object.

    Fields:
      additionalProperties: Additional properties of type SkewThresholdsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SkewThresholdsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1ThresholdConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1ThresholdConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributionScoreSkewThresholds = _messages.MessageField('AttributionScoreSkewThresholdsValue', 1)
    defaultSkewThreshold = _messages.MessageField('GoogleCloudAiplatformV1ThresholdConfig', 2)
    skewThresholds = _messages.MessageField('SkewThresholdsValue', 3)