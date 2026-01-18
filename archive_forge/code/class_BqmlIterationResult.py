from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BqmlIterationResult(_messages.Message):
    """A BqmlIterationResult object.

  Fields:
    durationMs: Deprecated.
    evalLoss: Deprecated.
    index: Deprecated.
    learnRate: Deprecated.
    trainingLoss: Deprecated.
  """
    durationMs = _messages.IntegerField(1)
    evalLoss = _messages.FloatField(2)
    index = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    learnRate = _messages.FloatField(4)
    trainingLoss = _messages.FloatField(5)