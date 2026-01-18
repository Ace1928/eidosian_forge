import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _post_epoch_days_before_year(year):
    """Get the number of days between 1970-01-01 and YEAR-01-01"""
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400 - EPOCHORDINAL