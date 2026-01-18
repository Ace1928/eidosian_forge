from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigQueryModelTraining(_messages.Message):
    """A BigQueryModelTraining object.

  Fields:
    currentIteration: Deprecated.
    expectedTotalIterations: Deprecated.
  """
    currentIteration = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    expectedTotalIterations = _messages.IntegerField(2)