from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
def _naive_isdst(self, dt, transitions):
    dston, dstoff = transitions
    dt = dt.replace(tzinfo=None)
    if dston < dstoff:
        isdst = dston <= dt < dstoff
    else:
        isdst = not dstoff <= dt < dston
    return isdst