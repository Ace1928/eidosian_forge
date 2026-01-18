from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfig(_messages.Message):
    """Configuration of how features in Featurestore are monitored.

  Fields:
    categoricalThresholdConfig: Threshold for categorical features of anomaly
      detection. This is shared by all types of Featurestore Monitoring for
      categorical features (i.e. Features with type (Feature.ValueType) BOOL
      or STRING).
    importFeaturesAnalysis: The config for ImportFeatures Analysis Based
      Feature Monitoring.
    numericalThresholdConfig: Threshold for numerical features of anomaly
      detection. This is shared by all objectives of Featurestore Monitoring
      for numerical features (i.e. Features with type (Feature.ValueType)
      DOUBLE or INT64).
    snapshotAnalysis: The config for Snapshot Analysis Based Feature
      Monitoring.
  """
    categoricalThresholdConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfigThresholdConfig', 1)
    importFeaturesAnalysis = _messages.MessageField('GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfigImportFeaturesAnalysis', 2)
    numericalThresholdConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfigThresholdConfig', 3)
    snapshotAnalysis = _messages.MessageField('GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfigSnapshotAnalysis', 4)