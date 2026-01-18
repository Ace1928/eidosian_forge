import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _get_local_timestamp(self, dt):
    return (dt.toordinal() - EPOCHORDINAL) * 86400 + dt.hour * 3600 + dt.minute * 60 + dt.second