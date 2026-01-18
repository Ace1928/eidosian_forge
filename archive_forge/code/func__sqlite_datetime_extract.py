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
def _sqlite_datetime_extract(lookup_type, dt, tzname=None, conn_tzname=None):
    dt = _sqlite_datetime_parse(dt, tzname, conn_tzname)
    if dt is None:
        return None
    if lookup_type == 'week_day':
        return dt.isoweekday() % 7 + 1
    elif lookup_type == 'iso_week_day':
        return dt.isoweekday()
    elif lookup_type == 'week':
        return dt.isocalendar().week
    elif lookup_type == 'quarter':
        return ceil(dt.month / 3)
    elif lookup_type == 'iso_year':
        return dt.isocalendar().year
    else:
        return getattr(dt, lookup_type)