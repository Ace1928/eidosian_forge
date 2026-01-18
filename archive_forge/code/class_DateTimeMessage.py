import datetime
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class DateTimeMessage(messages.Message):
    """Message to store/transmit a DateTime.

    Fields:
      milliseconds: Milliseconds since Jan 1st 1970 local time.
      time_zone_offset: Optional time zone offset, in minutes from UTC.
    """
    milliseconds = messages.IntegerField(1, required=True)
    time_zone_offset = messages.IntegerField(2)