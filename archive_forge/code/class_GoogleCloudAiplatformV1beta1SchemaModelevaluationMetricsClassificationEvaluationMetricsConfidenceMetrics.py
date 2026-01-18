from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsClassificationEvaluationMetricsConfidenceMetrics(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsClassification
  EvaluationMetricsConfidenceMetrics object.

  Fields:
    confidenceThreshold: Metrics are computed with an assumption that the
      Model never returns predictions with score lower than this value.
    confusionMatrix: Confusion matrix of the evaluation for this
      confidence_threshold.
    f1Score: The harmonic mean of recall and precision. For summary metrics,
      it computes the micro-averaged F1 score.
    f1ScoreAt1: The harmonic mean of recallAt1 and precisionAt1.
    f1ScoreMacro: Macro-averaged F1 Score.
    f1ScoreMicro: Micro-averaged F1 Score.
    falseNegativeCount: The number of ground truth labels that are not matched
      by a Model created label.
    falsePositiveCount: The number of Model created labels that do not match a
      ground truth label.
    falsePositiveRate: False Positive Rate for the given confidence threshold.
    falsePositiveRateAt1: The False Positive Rate when only considering the
      label that has the highest prediction score and not below the confidence
      threshold for each DataItem.
    maxPredictions: Metrics are computed with an assumption that the Model
      always returns at most this many predictions (ordered by their score,
      descendingly), but they all still need to meet the
      `confidenceThreshold`.
    precision: Precision for the given confidence threshold.
    precisionAt1: The precision when only considering the label that has the
      highest prediction score and not below the confidence threshold for each
      DataItem.
    recall: Recall (True Positive Rate) for the given confidence threshold.
    recallAt1: The Recall (True Positive Rate) when only considering the label
      that has the highest prediction score and not below the confidence
      threshold for each DataItem.
    trueNegativeCount: The number of labels that were not created by the
      Model, but if they would, they would not match a ground truth label.
    truePositiveCount: The number of Model created labels that match a ground
      truth label.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    confusionMatrix = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix', 2)
    f1Score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    f1ScoreAt1 = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    f1ScoreMacro = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    f1ScoreMicro = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    falseNegativeCount = _messages.IntegerField(7)
    falsePositiveCount = _messages.IntegerField(8)
    falsePositiveRate = _messages.FloatField(9, variant=_messages.Variant.FLOAT)
    falsePositiveRateAt1 = _messages.FloatField(10, variant=_messages.Variant.FLOAT)
    maxPredictions = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    precision = _messages.FloatField(12, variant=_messages.Variant.FLOAT)
    precisionAt1 = _messages.FloatField(13, variant=_messages.Variant.FLOAT)
    recall = _messages.FloatField(14, variant=_messages.Variant.FLOAT)
    recallAt1 = _messages.FloatField(15, variant=_messages.Variant.FLOAT)
    trueNegativeCount = _messages.IntegerField(16)
    truePositiveCount = _messages.IntegerField(17)