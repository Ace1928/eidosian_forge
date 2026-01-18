from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsForecastingEvaluationMetricsQuantileMetricsEntry(_messages.Message):
    """Entry for the Quantiles loss type optimization objective.

  Fields:
    observedQuantile: This is a custom metric that calculates the percentage
      of true values that were less than the predicted value for that
      quantile. Only populated when optimization_objective is minimize-
      quantile-loss and each entry corresponds to an entry in quantiles The
      percent value can be used to compare with the quantile value, which is
      the target value.
    quantile: The quantile for this entry.
    scaledPinballLoss: The scaled pinball loss of this quantile.
  """
    observedQuantile = _messages.FloatField(1)
    quantile = _messages.FloatField(2)
    scaledPinballLoss = _messages.FloatField(3, variant=_messages.Variant.FLOAT)