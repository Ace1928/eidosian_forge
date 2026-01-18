from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1EvaluationMetrics(_messages.Message):
    """Evaluation metrics, either in aggregate or about a specific entity.

  Fields:
    f1Score: The calculated f1 score.
    falseNegativesCount: The amount of false negatives.
    falsePositivesCount: The amount of false positives.
    groundTruthDocumentCount: The amount of documents with a ground truth
      occurrence.
    groundTruthOccurrencesCount: The amount of occurrences in ground truth
      documents.
    precision: The calculated precision.
    predictedDocumentCount: The amount of documents with a predicted
      occurrence.
    predictedOccurrencesCount: The amount of occurrences in predicted
      documents.
    recall: The calculated recall.
    totalDocumentsCount: The amount of documents that had an occurrence of
      this label.
    truePositivesCount: The amount of true positives.
  """
    f1Score = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    falseNegativesCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    falsePositivesCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    groundTruthDocumentCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    groundTruthOccurrencesCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    precision = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    predictedDocumentCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    predictedOccurrencesCount = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    recall = _messages.FloatField(9, variant=_messages.Variant.FLOAT)
    totalDocumentsCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    truePositivesCount = _messages.IntegerField(11, variant=_messages.Variant.INT32)