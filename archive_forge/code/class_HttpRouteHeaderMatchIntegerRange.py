from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteHeaderMatchIntegerRange(_messages.Message):
    """Represents an integer value range.

  Fields:
    end: End of the range (exclusive)
    start: Start of the range (inclusive)
  """
    end = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    start = _messages.IntegerField(2, variant=_messages.Variant.INT32)