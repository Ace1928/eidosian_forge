import time as _time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone, tzinfo
class LocalTimezoneOffsetWithUTC(tzinfo):
    """
    This is not intended to be a general purpose class for dealing with the
    local timezone. In particular:

    * it assumes that the date passed has been created using
      `datetime(..., tzinfo=Local)`, where `Local` is an instance of
      the object `LocalTimezoneOffsetWithUTC`;
    * for such an object, it returns the offset with UTC, used for date comparisons.

    Reference: https://docs.python.org/3/library/datetime.html
    """
    STDOFFSET = timedelta(seconds=-_time.timezone)
    if _time.daylight:
        DSTOFFSET = timedelta(seconds=-_time.altzone)
    else:
        DSTOFFSET = STDOFFSET

    def utcoffset(self, dt):
        """
        Access the relevant time offset.
        """
        return self.DSTOFFSET