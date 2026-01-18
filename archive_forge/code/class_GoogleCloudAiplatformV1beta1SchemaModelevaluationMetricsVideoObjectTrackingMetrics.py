from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsVideoObjectTrackingMetrics(_messages.Message):
    """Model evaluation metrics for video object tracking problems. Evaluates
  prediction quality of both labeled bounding boxes and labeled tracks (i.e.
  series of bounding boxes sharing same label and instance ID).

  Fields:
    boundingBoxMeanAveragePrecision: The single metric for bounding boxes
      evaluation: the `meanAveragePrecision` averaged over all
      `boundingBoxMetrics`.
    boundingBoxMetrics: The bounding boxes match metrics for each
      intersection-over-union threshold 0.05,0.10,...,0.95,0.96,0.97,0.98,0.99
      and each label confidence threshold
      0.05,0.10,...,0.95,0.96,0.97,0.98,0.99 pair.
    evaluatedBoundingBoxCount: UNIMPLEMENTED. The total number of bounding
      boxes (i.e. summed over all frames) the ground truth used to create this
      evaluation had.
    evaluatedFrameCount: UNIMPLEMENTED. The number of video frames used to
      create this evaluation.
    evaluatedTrackCount: UNIMPLEMENTED. The total number of tracks (i.e. as
      seen across all frames) the ground truth used to create this evaluation
      had.
    trackMeanAveragePrecision: UNIMPLEMENTED. The single metric for tracks
      accuracy evaluation: the `meanAveragePrecision` averaged over all
      `trackMetrics`.
    trackMeanBoundingBoxIou: UNIMPLEMENTED. The single metric for tracks
      bounding box iou evaluation: the `meanBoundingBoxIou` averaged over all
      `trackMetrics`.
    trackMeanMismatchRate: UNIMPLEMENTED. The single metric for tracking
      consistency evaluation: the `meanMismatchRate` averaged over all
      `trackMetrics`.
    trackMetrics: UNIMPLEMENTED. The tracks match metrics for each
      intersection-over-union threshold 0.05,0.10,...,0.95,0.96,0.97,0.98,0.99
      and each label confidence threshold
      0.05,0.10,...,0.95,0.96,0.97,0.98,0.99 pair.
  """
    boundingBoxMeanAveragePrecision = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    boundingBoxMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsBoundingBoxMetrics', 2, repeated=True)
    evaluatedBoundingBoxCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    evaluatedFrameCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    evaluatedTrackCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    trackMeanAveragePrecision = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    trackMeanBoundingBoxIou = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    trackMeanMismatchRate = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    trackMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTrackMetrics', 9, repeated=True)