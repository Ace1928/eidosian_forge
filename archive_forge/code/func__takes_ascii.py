from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz
from functools import wraps
import re
import six
def _takes_ascii(f):

    @wraps(f)
    def func(self, str_in, *args, **kwargs):
        str_in = getattr(str_in, 'read', lambda: str_in)()
        if isinstance(str_in, six.text_type):
            try:
                str_in = str_in.encode('ascii')
            except UnicodeEncodeError as e:
                msg = 'ISO-8601 strings should contain only ASCII characters'
                six.raise_from(ValueError(msg), e)
        return f(self, str_in, *args, **kwargs)
    return func