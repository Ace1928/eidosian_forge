from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class ITime(ITimeClass):
    """Represent time with time zone.

    Implemented by `datetime.time`.

    Operators:

    __repr__, __str__
    __cmp__, __hash__
    """
    hour = Attribute('Hour in range(24)')
    minute = Attribute('Minute in range(60)')
    second = Attribute('Second in range(60)')
    microsecond = Attribute('Microsecond in range(1000000)')
    tzinfo = Attribute('The object passed as the tzinfo argument to the time constructor\n        or None if none was passed.')

    def replace(hour, minute, second, microsecond, tzinfo):
        """Return a time with the same value.

        Except for those members given new values by whichever keyword
        arguments are specified. Note that tzinfo=None can be specified
        to create a naive time from an aware time, without conversion of the
        time members.
        """

    def isoformat():
        """Return a string representing the time in ISO 8601 format.

        That is HH:MM:SS.mmmmmm or, if self.microsecond is 0, HH:MM:SS
        If utcoffset() does not return None, a 6-character string is appended,
        giving the UTC offset in (signed) hours and minutes:
        HH:MM:SS.mmmmmm+HH:MM or, if self.microsecond is 0, HH:MM:SS+HH:MM
        """

    def __str__():
        """For a time t, str(t) is equivalent to t.isoformat()."""

    def strftime(format):
        """Return a string representing the time.

        This is controlled by an explicit format string.
        """

    def utcoffset():
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC).

        If tzinfo is None, returns None, else returns
        self.tzinfo.utcoffset(None), and raises an exception if the latter
        doesn't return None or a timedelta object representing a whole number
        of minutes with magnitude less than one day.
        """

    def dst():
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.

        If tzinfo is None, returns None, else returns self.tzinfo.dst(None),
        and raises an exception if the latter doesn't return None, or a
        timedelta object representing a whole number of minutes with
        magnitude less than one day.
        """

    def tzname():
        """Return the timezone name.

        If tzinfo is None, returns None, else returns self.tzinfo.tzname(None),
        or raises an exception if the latter doesn't return None or a string
        object.
        """