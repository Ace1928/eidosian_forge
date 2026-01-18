import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
class BaseTimestamp(datetime.datetime):
    """Our kind of wrapper over datetime.datetime.

  The objects produced by methods now, today, fromtimestamp, utcnow,
  utcfromtimestamp are timezone-aware (with correct timezone).
  We also overload __add__ and __sub__ method, to fix the result of arithmetic
  operations.
  """
    LocalTimezone = LocalTimezone

    @classmethod
    def AddLocalTimezone(cls, obj):
        """If obj is naive, add local timezone to it."""
        if not obj.tzinfo:
            return obj.replace(tzinfo=cls.LocalTimezone)
        return obj

    @classmethod
    def Localize(cls, obj):
        """If obj is naive, localize it to cls.LocalTimezone."""
        if not obj.tzinfo:
            return cls.LocalTimezone.localize(obj)
        return obj

    def __add__(self, *args, **kwargs):
        """x.__add__(y) <==> x+y."""
        r = super(BaseTimestamp, self).__add__(*args, **kwargs)
        return type(self)(r.year, r.month, r.day, r.hour, r.minute, r.second, r.microsecond, r.tzinfo)

    def __sub__(self, *args, **kwargs):
        """x.__add__(y) <==> x-y."""
        r = super(BaseTimestamp, self).__sub__(*args, **kwargs)
        if isinstance(r, datetime.datetime):
            return type(self)(r.year, r.month, r.day, r.hour, r.minute, r.second, r.microsecond, r.tzinfo)
        return r

    @classmethod
    def now(cls, *args, **kwargs):
        """Get a timestamp corresponding to right now.

    Args:
      args: Positional arguments to pass to datetime.datetime.now().
      kwargs: Keyword arguments to pass to datetime.datetime.now(). If tz is not
              specified, local timezone is assumed.

    Returns:
      A new BaseTimestamp with tz's local day and time.
    """
        return cls.AddLocalTimezone(super(BaseTimestamp, cls).now(*args, **kwargs))

    @classmethod
    def today(cls):
        """Current BaseTimestamp.

    Same as self.__class__.fromtimestamp(time.time()).
    Returns:
      New self.__class__.
    """
        return cls.AddLocalTimezone(super(BaseTimestamp, cls).today())

    @classmethod
    def fromtimestamp(cls, *args, **kwargs):
        """Get a new localized timestamp from a POSIX timestamp.

    Args:
      args: Positional arguments to pass to datetime.datetime.fromtimestamp().
      kwargs: Keyword arguments to pass to datetime.datetime.fromtimestamp().
              If tz is not specified, local timezone is assumed.

    Returns:
      A new BaseTimestamp with tz's local day and time.
    """
        return cls.Localize(super(BaseTimestamp, cls).fromtimestamp(*args, **kwargs))

    @classmethod
    def utcnow(cls):
        """Return a new BaseTimestamp representing UTC day and time."""
        return super(BaseTimestamp, cls).utcnow().replace(tzinfo=pytz.utc)

    @classmethod
    def utcfromtimestamp(cls, *args, **kwargs):
        """timestamp -> UTC datetime from a POSIX timestamp (like time.time())."""
        return super(BaseTimestamp, cls).utcfromtimestamp(*args, **kwargs).replace(tzinfo=pytz.utc)

    @classmethod
    def strptime(cls, date_string, format, tz=None):
        """Parse date_string according to format and construct BaseTimestamp.

    Args:
      date_string: string passed to time.strptime.
      format: format string passed to time.strptime.
      tz: if not specified, local timezone assumed.
    Returns:
      New BaseTimestamp.
    """
        if tz is None:
            return cls.Localize(cls(*time.strptime(date_string, format)[:6]))
        return tz.localize(cls(*time.strptime(date_string, format)[:6]))

    def astimezone(self, *args, **kwargs):
        """tz -> convert to time in new timezone tz."""
        r = super(BaseTimestamp, self).astimezone(*args, **kwargs)
        return type(self)(r.year, r.month, r.day, r.hour, r.minute, r.second, r.microsecond, r.tzinfo)

    @classmethod
    def FromMicroTimestamp(cls, ts):
        """Create new Timestamp object from microsecond UTC timestamp value.

    Args:
      ts: integer microsecond UTC timestamp
    Returns:
      New cls()
    """
        return cls.utcfromtimestamp(ts / _MICROSECONDS_PER_SECOND_F)

    def AsSecondsSinceEpoch(self):
        """Return number of seconds since epoch (timestamp in seconds)."""
        return GetSecondsSinceEpoch(self.utctimetuple())

    def AsMicroTimestamp(self):
        """Return microsecond timestamp constructed from this object."""
        return SecondsToMicroseconds(self.AsSecondsSinceEpoch()) + self.microsecond

    @classmethod
    def combine(cls, datepart, timepart, tz=None):
        """Combine date and time into timestamp, timezone-aware.

    Args:
      datepart: datetime.date
      timepart: datetime.time
      tz: timezone or None
    Returns:
      timestamp object
    """
        result = super(BaseTimestamp, cls).combine(datepart, timepart)
        if tz:
            result = tz.localize(result)
        return result