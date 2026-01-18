from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTrackMetricsConfidenceMetrics(_messages.Message):
    """Metrics for a single confidence threshold.

  Fields:
    boundingBoxIou: Bounding box intersection-over-union precision. Measures
      how well the bounding boxes overlap between each other (e.g. complete
      overlap or just barely above iou_threshold).
    confidenceThreshold: The confidence threshold value used to compute the
      metrics.
    mismatchRate: Mismatch rate, which measures the tracking consistency, i.e.
      correctness of instance ID continuity.
    trackingPrecision: Tracking precision.
    trackingRecall: Tracking recall.
  """
    boundingBoxIou = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    confidenceThreshold = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    mismatchRate = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    trackingPrecision = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    trackingRecall = _messages.FloatField(5, variant=_messages.Variant.FLOAT)