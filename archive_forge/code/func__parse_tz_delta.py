import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _parse_tz_delta(tz_delta):
    match = re.fullmatch('(?P<sign>[+-])?(?P<h>\\d{1,3})(:(?P<m>\\d{2})(:(?P<s>\\d{2}))?)?', tz_delta, re.ASCII)
    assert match is not None, tz_delta
    h, m, s = (int(v or 0) for v in match.group('h', 'm', 's'))
    total = h * 3600 + m * 60 + s
    if h > 24:
        raise ValueError(f'Offset hours must be in [0, 24]: {tz_delta}')
    if match.group('sign') != '-':
        total = -total
    return total