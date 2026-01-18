from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
def _parse_isodate_uncommon(self, dt_str):
    if len(dt_str) < 4:
        raise ValueError('ISO string too short')
    year = int(dt_str[0:4])
    has_sep = dt_str[4:5] == self._DATE_SEP
    pos = 4 + has_sep
    if dt_str[pos:pos + 1] == b'W':
        pos += 1
        weekno = int(dt_str[pos:pos + 2])
        pos += 2
        dayno = 1
        if len(dt_str) > pos:
            if (dt_str[pos:pos + 1] == self._DATE_SEP) != has_sep:
                raise ValueError('Inconsistent use of dash separator')
            pos += has_sep
            dayno = int(dt_str[pos:pos + 1])
            pos += 1
        base_date = self._calculate_weekdate(year, weekno, dayno)
    else:
        if len(dt_str) - pos < 3:
            raise ValueError('Invalid ordinal day')
        ordinal_day = int(dt_str[pos:pos + 3])
        pos += 3
        if ordinal_day < 1 or ordinal_day > 365 + calendar.isleap(year):
            raise ValueError('Invalid ordinal day' + ' {} for year {}'.format(ordinal_day, year))
        base_date = date(year, 1, 1) + timedelta(days=ordinal_day - 1)
    components = [base_date.year, base_date.month, base_date.day]
    return (components, pos)