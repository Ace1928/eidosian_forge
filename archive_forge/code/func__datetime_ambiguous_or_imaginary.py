import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
def _datetime_ambiguous_or_imaginary(dt, tz):
    return tz.utcoffset(dt.replace(fold=not dt.fold)) != tz.utcoffset(dt)