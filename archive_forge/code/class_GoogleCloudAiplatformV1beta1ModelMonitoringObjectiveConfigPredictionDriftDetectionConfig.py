from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfigPredictionDriftDetectionConfig(_messages.Message):
    """The config for Prediction data drift detection.

  Messages:
    AttributionScoreDriftThresholdsValue: Key is the feature name and value is
      the threshold. The threshold here is against attribution score distance
      between different time windows.
    DriftThresholdsValue: Key is the feature name and value is the threshold.
      If a feature needs to be monitored for drift, a value threshold must be
      configured for that feature. The threshold here is against feature
      distribution distance between different time windws.

  Fields:
    attributionScoreDriftThresholds: Key is the feature name and value is the
      threshold. The threshold here is against attribution score distance
      between different time windows.
    defaultDriftThreshold: Drift anomaly detection threshold used by all
      features. When the per-feature thresholds are not set, this field can be
      used to specify a threshold for all features.
    driftThresholds: Key is the feature name and value is the threshold. If a
      feature needs to be monitored for drift, a value threshold must be
      configured for that feature. The threshold here is against feature
      distribution distance between different time windws.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributionScoreDriftThresholdsValue(_messages.Message):
        """Key is the feature name and value is the threshold. The threshold here
    is against attribution score distance between different time windows.

    Messages:
      AdditionalProperty: An additional property for a
        AttributionScoreDriftThresholdsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AttributionScoreDriftThresholdsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributionScoreDriftThresholdsValue
      object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1ThresholdConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1ThresholdConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DriftThresholdsValue(_messages.Message):
        """Key is the feature name and value is the threshold. If a feature needs
    to be monitored for drift, a value threshold must be configured for that
    feature. The threshold here is against feature distribution distance
    between different time windws.

    Messages:
      AdditionalProperty: An additional property for a DriftThresholdsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DriftThresholdsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DriftThresholdsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1ThresholdConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1ThresholdConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributionScoreDriftThresholds = _messages.MessageField('AttributionScoreDriftThresholdsValue', 1)
    defaultDriftThreshold = _messages.MessageField('GoogleCloudAiplatformV1beta1ThresholdConfig', 2)
    driftThresholds = _messages.MessageField('DriftThresholdsValue', 3)