from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FloatingPointMean(_messages.Message):
    """A representation of a floating point mean metric contribution.

  Fields:
    count: The number of values being aggregated.
    sum: The sum of all values being aggregated.
  """
    count = _messages.MessageField('SplitInt64', 1)
    sum = _messages.FloatField(2)