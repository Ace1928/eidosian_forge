from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsClassificationEvaluationMetrics(_messages.Message):
    """Metrics for classification evaluation results.

  Fields:
    auPrc: The Area Under Precision-Recall Curve metric. Micro-averaged for
      the overall evaluation.
    auRoc: The Area Under Receiver Operating Characteristic curve metric.
      Micro-averaged for the overall evaluation.
    confidenceMetrics: Metrics for each `confidenceThreshold` in
      0.00,0.05,0.10,...,0.95,0.96,0.97,0.98,0.99 and `positionThreshold` =
      INT32_MAX_VALUE. ROC and precision-recall curves, and other aggregated
      metrics are derived from them. The confidence metrics entries may also
      be supplied for additional values of `positionThreshold`, but from these
      no aggregated metrics are computed.
    confusionMatrix: Confusion matrix of the evaluation.
    logLoss: The Log Loss metric.
  """
    auPrc = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    auRoc = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    confidenceMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsClassificationEvaluationMetricsConfidenceMetrics', 3, repeated=True)
    confusionMatrix = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix', 4)
    logLoss = _messages.FloatField(5, variant=_messages.Variant.FLOAT)