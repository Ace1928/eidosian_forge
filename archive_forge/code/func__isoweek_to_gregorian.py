import time as _time
import math as _math
import sys
from operator import index as _index
def _isoweek_to_gregorian(year, week, day):
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError(f'Year is out of range: {year}')
    if not 0 < week < 53:
        out_of_range = True
        if week == 53:
            first_weekday = _ymd2ord(year, 1, 1) % 7
            if first_weekday == 4 or (first_weekday == 3 and _is_leap(year)):
                out_of_range = False
        if out_of_range:
            raise ValueError(f'Invalid week: {week}')
    if not 0 < day < 8:
        raise ValueError(f'Invalid weekday: {day} (range is [1, 7])')
    day_offset = (week - 1) * 7 + (day - 1)
    day_1 = _isoweek1monday(year)
    ord_day = day_1 + day_offset
    return _ord2ymd(ord_day)