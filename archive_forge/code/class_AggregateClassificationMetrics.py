from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregateClassificationMetrics(_messages.Message):
    """Aggregate metrics for classification/classifier models. For multi-class
  models, the metrics are either macro-averaged or micro-averaged. When macro-
  averaged, the metrics are calculated for each label and then an unweighted
  average is taken of those values. When micro-averaged, the metric is
  calculated globally by counting the total number of correctly predicted
  rows.

  Fields:
    accuracy: Accuracy is the fraction of predictions given the correct label.
      For multiclass this is a micro-averaged metric.
    f1Score: The F1 score is an average of recall and precision. For
      multiclass this is a macro-averaged metric.
    logLoss: Logarithmic Loss. For multiclass this is a macro-averaged metric.
    precision: Precision is the fraction of actual positive predictions that
      had positive actual labels. For multiclass this is a macro-averaged
      metric treating each class as a binary classifier.
    recall: Recall is the fraction of actual positive labels that were given a
      positive prediction. For multiclass this is a macro-averaged metric.
    rocAuc: Area Under a ROC Curve. For multiclass this is a macro-averaged
      metric.
    threshold: Threshold at which the metrics are computed. For binary
      classification models this is the positive class threshold. For multi-
      class classfication models this is the confidence threshold.
  """
    accuracy = _messages.FloatField(1)
    f1Score = _messages.FloatField(2)
    logLoss = _messages.FloatField(3)
    precision = _messages.FloatField(4)
    recall = _messages.FloatField(5)
    rocAuc = _messages.FloatField(6)
    threshold = _messages.FloatField(7)