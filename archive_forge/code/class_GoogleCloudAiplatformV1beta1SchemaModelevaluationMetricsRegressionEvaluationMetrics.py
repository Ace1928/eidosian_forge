from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsRegressionEvaluationMetrics(_messages.Message):
    """Metrics for regression evaluation results.

  Fields:
    meanAbsoluteError: Mean Absolute Error (MAE).
    meanAbsolutePercentageError: Mean absolute percentage error. Infinity when
      there are zeros in the ground truth.
    rSquared: Coefficient of determination as Pearson correlation coefficient.
      Undefined when ground truth or predictions are constant or near
      constant.
    rootMeanSquaredError: Root Mean Squared Error (RMSE).
    rootMeanSquaredLogError: Root mean squared log error. Undefined when there
      are negative ground truth values or predictions.
  """
    meanAbsoluteError = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    meanAbsolutePercentageError = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    rSquared = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    rootMeanSquaredError = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    rootMeanSquaredLogError = _messages.FloatField(5, variant=_messages.Variant.FLOAT)