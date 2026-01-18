import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def _datetime_to_json(dt):
    if isinstance(dt, datetime.datetime):
        r = dt.isoformat()
        if dt.microsecond:
            r = r[:23] + r[26:]
        if r.endswith('+00:00'):
            r = r[:-6] + 'Z'
        return r
    elif isinstance(dt, datetime.time):
        r = dt.isoformat()
        if dt.microsecond:
            r = r[:12]
        return r
    else:
        return dt.isoformat()