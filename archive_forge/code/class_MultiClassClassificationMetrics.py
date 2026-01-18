from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiClassClassificationMetrics(_messages.Message):
    """Evaluation metrics for multi-class classification/classifier models.

  Fields:
    aggregateClassificationMetrics: Aggregate classification metrics.
    confusionMatrixList: Confusion matrix at different thresholds.
  """
    aggregateClassificationMetrics = _messages.MessageField('AggregateClassificationMetrics', 1)
    confusionMatrixList = _messages.MessageField('ConfusionMatrix', 2, repeated=True)