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
def _sqlite_datetime_parse(dt, tzname=None, conn_tzname=None):
    if dt is None:
        return None
    try:
        dt = typecast_timestamp(dt)
    except (TypeError, ValueError):
        return None
    if conn_tzname:
        dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(conn_tzname))
    if tzname is not None and tzname != conn_tzname:
        tzname, sign, offset = split_tzname_delta(tzname)
        if offset:
            hours, minutes = offset.split(':')
            offset_delta = timedelta(hours=int(hours), minutes=int(minutes))
            dt += offset_delta if sign == '+' else -offset_delta
        dt = timezone.localtime(dt, zoneinfo.ZoneInfo(tzname))
    return dt