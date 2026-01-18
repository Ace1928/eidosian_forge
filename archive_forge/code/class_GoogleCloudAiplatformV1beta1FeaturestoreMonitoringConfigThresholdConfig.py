from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfigThresholdConfig(_messages.Message):
    """The config for Featurestore Monitoring threshold.

  Fields:
    value: Specify a threshold value that can trigger the alert. 1. For
      categorical feature, the distribution distance is calculated by
      L-inifinity norm. 2. For numerical feature, the distribution distance is
      calculated by Jensen\\u2013Shannon divergence. Each feature must have a
      non-zero threshold if they need to be monitored. Otherwise no alert will
      be triggered for that feature.
  """
    value = _messages.FloatField(1)