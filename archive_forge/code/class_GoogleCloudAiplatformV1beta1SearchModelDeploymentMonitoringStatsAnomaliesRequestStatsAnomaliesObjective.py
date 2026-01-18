from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SearchModelDeploymentMonitoringStatsAnomaliesRequestStatsAnomaliesObjective(_messages.Message):
    """Stats requested for specific objective.

  Enums:
    TypeValueValuesEnum:

  Fields:
    topFeatureCount: If set, all attribution scores between
      SearchModelDeploymentMonitoringStatsAnomaliesRequest.start_time and
      SearchModelDeploymentMonitoringStatsAnomaliesRequest.end_time are
      fetched, and page token doesn't take effect in this case. Only used to
      retrieve attribution score for the top Features which has the highest
      attribution score in the latest monitoring run.
    type: A TypeValueValuesEnum attribute.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """TypeValueValuesEnum enum type.

    Values:
      MODEL_DEPLOYMENT_MONITORING_OBJECTIVE_TYPE_UNSPECIFIED: Default value,
        should not be set.
      RAW_FEATURE_SKEW: Raw feature values' stats to detect skew between
        Training-Prediction datasets.
      RAW_FEATURE_DRIFT: Raw feature values' stats to detect drift between
        Serving-Prediction datasets.
      FEATURE_ATTRIBUTION_SKEW: Feature attribution scores to detect skew
        between Training-Prediction datasets.
      FEATURE_ATTRIBUTION_DRIFT: Feature attribution scores to detect skew
        between Prediction datasets collected within different time windows.
    """
        MODEL_DEPLOYMENT_MONITORING_OBJECTIVE_TYPE_UNSPECIFIED = 0
        RAW_FEATURE_SKEW = 1
        RAW_FEATURE_DRIFT = 2
        FEATURE_ATTRIBUTION_SKEW = 3
        FEATURE_ATTRIBUTION_DRIFT = 4
    topFeatureCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 2)