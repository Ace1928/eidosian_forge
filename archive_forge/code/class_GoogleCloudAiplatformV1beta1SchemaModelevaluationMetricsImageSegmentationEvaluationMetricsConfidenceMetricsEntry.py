from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsImageSegmentationEvaluationMetricsConfidenceMetricsEntry(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsImageSegmentat
  ionEvaluationMetricsConfidenceMetricsEntry object.

  Fields:
    confidenceThreshold: Metrics are computed with an assumption that the
      model never returns predictions with score lower than this value.
    confusionMatrix: Confusion matrix for the given confidence threshold.
    diceScoreCoefficient: DSC or the F1 score, The harmonic mean of recall and
      precision.
    iouScore: The intersection-over-union score. The measure of overlap of the
      annotation's category mask with ground truth category mask on the
      DataItem.
    precision: Precision for the given confidence threshold.
    recall: Recall (True Positive Rate) for the given confidence threshold.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    confusionMatrix = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix', 2)
    diceScoreCoefficient = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    iouScore = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    precision = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    recall = _messages.FloatField(6, variant=_messages.Variant.FLOAT)