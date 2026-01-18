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
def _sqlite_format_dtdelta(connector, lhs, rhs):
    """
    LHS and RHS can be either:
    - An integer number of microseconds
    - A string representing a datetime
    - A scalar value, e.g. float
    """
    if connector is None or lhs is None or rhs is None:
        return None
    connector = connector.strip()
    try:
        real_lhs = _sqlite_prepare_dtdelta_param(connector, lhs)
        real_rhs = _sqlite_prepare_dtdelta_param(connector, rhs)
    except (ValueError, TypeError):
        return None
    if connector == '+':
        out = str(real_lhs + real_rhs)
    elif connector == '-':
        out = str(real_lhs - real_rhs)
    elif connector == '*':
        out = real_lhs * real_rhs
    else:
        out = real_lhs / real_rhs
    return out