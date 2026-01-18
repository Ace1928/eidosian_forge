import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def FromDatetime(self, dt):
    """Converts datetime to Timestamp.

    Args:
      dt: A datetime. If it's timezone-naive, it's assumed to be in UTC.
    """
    seconds = calendar.timegm(dt.utctimetuple())
    nanos = dt.microsecond * _NANOS_PER_MICROSECOND
    _CheckTimestampValid(seconds, nanos)
    self.seconds = seconds
    self.nanos = nanos