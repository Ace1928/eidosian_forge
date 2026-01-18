from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
def _parse_isotime(self, timestr):
    len_str = len(timestr)
    components = [0, 0, 0, 0, None]
    pos = 0
    comp = -1
    if len_str < 2:
        raise ValueError('ISO time too short')
    has_sep = False
    while pos < len_str and comp < 5:
        comp += 1
        if timestr[pos:pos + 1] in b'-+Zz':
            components[-1] = self._parse_tzstr(timestr[pos:])
            pos = len_str
            break
        if comp == 1 and timestr[pos:pos + 1] == self._TIME_SEP:
            has_sep = True
            pos += 1
        elif comp == 2 and has_sep:
            if timestr[pos:pos + 1] != self._TIME_SEP:
                raise ValueError('Inconsistent use of colon separator')
            pos += 1
        if comp < 3:
            components[comp] = int(timestr[pos:pos + 2])
            pos += 2
        if comp == 3:
            frac = self._FRACTION_REGEX.match(timestr[pos:])
            if not frac:
                continue
            us_str = frac.group(1)[:6]
            components[comp] = int(us_str) * 10 ** (6 - len(us_str))
            pos += len(frac.group())
    if pos < len_str:
        raise ValueError('Unused components in ISO string')
    if components[0] == 24:
        if any((component != 0 for component in components[1:4])):
            raise ValueError('Hour may only be 24 at 24:00:00.000')
    return components