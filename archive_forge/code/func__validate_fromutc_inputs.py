from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
def _validate_fromutc_inputs(f):
    """
    The CPython version of ``fromutc`` checks that the input is a ``datetime``
    object and that ``self`` is attached as its ``tzinfo``.
    """

    @wraps(f)
    def fromutc(self, dt):
        if not isinstance(dt, datetime):
            raise TypeError('fromutc() requires a datetime argument')
        if dt.tzinfo is not self:
            raise ValueError('dt.tzinfo is not self')
        return f(self, dt)
    return fromutc