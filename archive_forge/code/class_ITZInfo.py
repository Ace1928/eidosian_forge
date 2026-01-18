from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class ITZInfo(Interface):
    """Time zone info class.
    """

    def utcoffset(dt):
        """Return offset of local time from UTC, in minutes east of UTC.

        If local time is west of UTC, this should be negative.
        Note that this is intended to be the total offset from UTC;
        for example, if a tzinfo object represents both time zone and DST
        adjustments, utcoffset() should return their sum. If the UTC offset
        isn't known, return None. Else the value returned must be a timedelta
        object specifying a whole number of minutes in the range -1439 to 1439
        inclusive (1440 = 24*60; the magnitude of the offset must be less
        than one day).
        """

    def dst(dt):
        """Return the daylight saving time (DST) adjustment, in minutes east
        of UTC, or None if DST information isn't known.
        """

    def tzname(dt):
        """Return the time zone name corresponding to the datetime object as
        a string.
        """

    def fromutc(dt):
        """Return an equivalent datetime in self's local time."""