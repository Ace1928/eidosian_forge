from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectiveValueValuesEnum(_messages.Enum):
    """Model Monitoring Objective those stats and anomalies belonging to.

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