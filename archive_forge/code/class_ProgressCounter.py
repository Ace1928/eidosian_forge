from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProgressCounter(_messages.Message):
    """ProgressCounter provides counters to describe an operation's progress.

  Fields:
    failure: The number of units that failed in the operation.
    pending: The number of units that are pending in the operation.
    success: The number of units that succeeded in the operation.
  """
    failure = _messages.IntegerField(1)
    pending = _messages.IntegerField(2)
    success = _messages.IntegerField(3)