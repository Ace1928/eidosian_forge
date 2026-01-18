from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsVideoActionMetrics(_messages.Message):
    """The Evaluation metrics given a specific precision_window_length.

  Fields:
    confidenceMetrics: Metrics for each label-match confidence_threshold from
      0.05,0.10,...,0.95,0.96,0.97,0.98,0.99.
    meanAveragePrecision: The mean average precision.
    precisionWindowLength: This VideoActionMetrics is calculated based on this
      prediction window length. If the predicted action's timestamp is inside
      the time window whose center is the ground truth action's timestamp with
      this specific length, the prediction result is treated as a true
      positive.
  """
    confidenceMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsVideoActionMetricsConfidenceMetrics', 1, repeated=True)
    meanAveragePrecision = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    precisionWindowLength = _messages.StringField(3)