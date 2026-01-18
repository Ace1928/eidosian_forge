from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Int64RangeMatch(_messages.Message):
    """HttpRouteRuleMatch criteria for field values that must stay within the
  specified integer range.

  Fields:
    rangeEnd: The end of the range (exclusive) in signed long integer format.
    rangeStart: The start of the range (inclusive) in signed long integer
      format.
  """
    rangeEnd = _messages.IntegerField(1)
    rangeStart = _messages.IntegerField(2)