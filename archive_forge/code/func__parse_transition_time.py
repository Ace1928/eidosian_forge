import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _parse_transition_time(time_str):
    match = re.fullmatch('(?P<sign>[+-])?(?P<h>\\d{1,3})(:(?P<m>\\d{2})(:(?P<s>\\d{2}))?)?', time_str, re.ASCII)
    if match is None:
        raise ValueError(f'Invalid time: {time_str}')
    h, m, s = (int(v or 0) for v in match.group('h', 'm', 's'))
    if h > 167:
        raise ValueError(f'Hour must be in [0, 167]: {time_str}')
    if match.group('sign') == '-':
        h, m, s = (-h, -m, -s)
    return (h, m, s)