from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTextExtractionEvaluationMetricsConfidenceMetrics(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTextExtraction
  EvaluationMetricsConfidenceMetrics object.

  Fields:
    confidenceThreshold: Metrics are computed with an assumption that the
      Model never returns predictions with score lower than this value.
    f1Score: The harmonic mean of recall and precision.
    precision: Precision for the given confidence threshold.
    recall: Recall (True Positive Rate) for the given confidence threshold.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    f1Score = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    precision = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    recall = _messages.FloatField(4, variant=_messages.Variant.FLOAT)