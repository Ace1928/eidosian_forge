from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservationTimeWindow(_messages.Message):
    """A FutureReservationTimeWindow object.

  Fields:
    duration: A Duration attribute.
    endTime: A string attribute.
    startTime: Start time of the Future Reservation. The start_time is an
      RFC3339 string.
  """
    duration = _messages.MessageField('Duration', 1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)