from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import model_monitoring_jobs_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _ConstructDriftThresholds(self, feature_thresholds, feature_attribution_thresholds):
    """Construct drift thresholds from user input.

    Args:
      feature_thresholds: Dict or None, key: feature_name, value: thresholds.
      feature_attribution_thresholds: Dict or None, key:feature_name, value:
        attribution score thresholds.

    Returns:
      PredictionDriftDetectionConfig
    """
    prediction_drift_detection = api_util.GetMessage('ModelMonitoringObjectiveConfigPredictionDriftDetectionConfig', self._version)()
    additional_properties = []
    attribution_additional_properties = []
    if feature_thresholds:
        for key, value in feature_thresholds.items():
            threshold = 0.3 if not value else float(value)
            additional_properties.append(prediction_drift_detection.DriftThresholdsValue().AdditionalProperty(key=key, value=api_util.GetMessage('ThresholdConfig', self._version)(value=threshold)))
        prediction_drift_detection.driftThresholds = prediction_drift_detection.DriftThresholdsValue(additionalProperties=additional_properties)
    if feature_attribution_thresholds:
        for key, value in feature_attribution_thresholds.items():
            threshold = 0.3 if not value else float(value)
            attribution_additional_properties.append(prediction_drift_detection.AttributionScoreDriftThresholdsValue().AdditionalProperty(key=key, value=api_util.GetMessage('ThresholdConfig', self._version)(value=threshold)))
        prediction_drift_detection.attributionScoreDriftThresholds = prediction_drift_detection.AttributionScoreDriftThresholdsValue(additionalProperties=attribution_additional_properties)
    return prediction_drift_detection