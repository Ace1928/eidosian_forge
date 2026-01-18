import functools
import random
import statistics
import zoneinfo
from datetime import timedelta
from hashlib import md5, sha1, sha224, sha256, sha384, sha512
from math import (
from re import search as re_search
from django.db.backends.utils import (
from django.utils import timezone
from django.utils.duration import duration_microseconds
def _sqlite_time_trunc(lookup_type, dt, tzname, conn_tzname):
    if dt is None:
        return None
    dt_parsed = _sqlite_datetime_parse(dt, tzname, conn_tzname)
    if dt_parsed is None:
        try:
            dt = typecast_time(dt)
        except (ValueError, TypeError):
            return None
    else:
        dt = dt_parsed
    if lookup_type == 'hour':
        return f'{dt.hour:02d}:00:00'
    elif lookup_type == 'minute':
        return f'{dt.hour:02d}:{dt.minute:02d}:00'
    elif lookup_type == 'second':
        return f'{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}'
    raise ValueError(f'Unsupported lookup type: {lookup_type!r}')