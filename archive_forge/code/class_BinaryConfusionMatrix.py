from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryConfusionMatrix(_messages.Message):
    """Confusion matrix for binary classification models.

  Fields:
    accuracy: The fraction of predictions given the correct label.
    f1Score: The equally weighted average of recall and precision.
    falseNegatives: Number of false samples predicted as false.
    falsePositives: Number of false samples predicted as true.
    positiveClassThreshold: Threshold value used when computing each of the
      following metric.
    precision: The fraction of actual positive predictions that had positive
      actual labels.
    recall: The fraction of actual positive labels that were given a positive
      prediction.
    trueNegatives: Number of true samples predicted as false.
    truePositives: Number of true samples predicted as true.
  """
    accuracy = _messages.FloatField(1)
    f1Score = _messages.FloatField(2)
    falseNegatives = _messages.IntegerField(3)
    falsePositives = _messages.IntegerField(4)
    positiveClassThreshold = _messages.FloatField(5)
    precision = _messages.FloatField(6)
    recall = _messages.FloatField(7)
    trueNegatives = _messages.IntegerField(8)
    truePositives = _messages.IntegerField(9)