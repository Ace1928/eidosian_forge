from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryClassificationMetrics(_messages.Message):
    """Evaluation metrics for binary classification/classifier models.

  Fields:
    aggregateClassificationMetrics: Aggregate classification metrics.
    binaryConfusionMatrixList: Binary confusion matrix at multiple thresholds.
    negativeLabel: Label representing the negative class.
    positiveLabel: Label representing the positive class.
  """
    aggregateClassificationMetrics = _messages.MessageField('AggregateClassificationMetrics', 1)
    binaryConfusionMatrixList = _messages.MessageField('BinaryConfusionMatrix', 2, repeated=True)
    negativeLabel = _messages.StringField(3)
    positiveLabel = _messages.StringField(4)