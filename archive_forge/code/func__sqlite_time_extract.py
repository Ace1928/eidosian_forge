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
def _sqlite_time_extract(lookup_type, dt):
    if dt is None:
        return None
    try:
        dt = typecast_time(dt)
    except (ValueError, TypeError):
        return None
    return getattr(dt, lookup_type)