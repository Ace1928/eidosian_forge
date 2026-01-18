from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
def _parse_tzstr(self, tzstr, zero_as_utc=True):
    if tzstr == b'Z' or tzstr == b'z':
        return tz.UTC
    if len(tzstr) not in {3, 5, 6}:
        raise ValueError('Time zone offset must be 1, 3, 5 or 6 characters')
    if tzstr[0:1] == b'-':
        mult = -1
    elif tzstr[0:1] == b'+':
        mult = 1
    else:
        raise ValueError('Time zone offset requires sign')
    hours = int(tzstr[1:3])
    if len(tzstr) == 3:
        minutes = 0
    else:
        minutes = int(tzstr[4 if tzstr[3:4] == self._TIME_SEP else 3:])
    if zero_as_utc and hours == 0 and (minutes == 0):
        return tz.UTC
    else:
        if minutes > 59:
            raise ValueError('Invalid minutes in time zone offset')
        if hours > 23:
            raise ValueError('Invalid hours in time zone offset')
        return tz.tzoffset(None, mult * (hours * 60 + minutes) * 60)