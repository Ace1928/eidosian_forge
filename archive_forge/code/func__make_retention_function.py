import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
@staticmethod
def _make_retention_function(retention):
    if retention is None:
        return None
    elif isinstance(retention, str):
        interval = string_parsers.parse_duration(retention)
        if interval is None:
            raise ValueError("Cannot parse retention from: '%s'" % retention)
        return FileSink._make_retention_function(interval)
    elif isinstance(retention, int):
        return partial(Retention.retention_count, number=retention)
    elif isinstance(retention, datetime.timedelta):
        return partial(Retention.retention_age, seconds=retention.total_seconds())
    elif callable(retention):
        return retention
    else:
        raise TypeError("Cannot infer retention for objects of type: '%s'" % type(retention).__name__)