from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleTypeInterval(_messages.Message):
    """Represents a time interval, encoded as a Timestamp start (inclusive) and
  a Timestamp end (exclusive). The start must be less than or equal to the
  end. When the start equals the end, the interval is empty (matches no time).
  When both start and end are unspecified, the interval matches any time.

  Fields:
    endTime: Optional. Exclusive end of the interval. If specified, a
      Timestamp matching this interval will have to be before the end.
    startTime: Required. Inclusive start of the interval. If specified, a
      Timestamp matching this interval will have to be the same or after the
      start.
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)