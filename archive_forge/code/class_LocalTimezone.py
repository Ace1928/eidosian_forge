from __future__ import annotations
import numbers
import os
import random
import sys
import time as _time
from calendar import monthrange
from datetime import date, datetime, timedelta
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from types import ModuleType
from typing import Any, Callable
from dateutil import tz as dateutil_tz
from dateutil.parser import isoparse
from kombu.utils.functional import reprcall
from kombu.utils.objects import cached_property
from .functional import dictfilter
from .text import pluralize
class LocalTimezone(tzinfo):
    """Local time implementation. Provided in _Zone to the app when `enable_utc` is disabled.
    Otherwise, _Zone provides a UTC ZoneInfo instance as the timezone implementation for the application.

    Note:
        Used only when the :setting:`enable_utc` setting is disabled.
    """
    _offset_cache: dict[int, tzinfo] = {}

    def __init__(self) -> None:
        self.STDOFFSET = timedelta(seconds=-_time.timezone)
        if _time.daylight:
            self.DSTOFFSET = timedelta(seconds=-_time.altzone)
        else:
            self.DSTOFFSET = self.STDOFFSET
        self.DSTDIFF = self.DSTOFFSET - self.STDOFFSET
        super().__init__()

    def __repr__(self) -> str:
        return f'<LocalTimezone: UTC{int(self.DSTOFFSET.total_seconds() / 3600):+03d}>'

    def utcoffset(self, dt: datetime) -> timedelta:
        return self.DSTOFFSET if self._isdst(dt) else self.STDOFFSET

    def dst(self, dt: datetime) -> timedelta:
        return self.DSTDIFF if self._isdst(dt) else ZERO

    def tzname(self, dt: datetime) -> str:
        return _time.tzname[self._isdst(dt)]

    def fromutc(self, dt: datetime) -> datetime:
        offset = int(self.utcoffset(dt).seconds / 60.0)
        try:
            tz = self._offset_cache[offset]
        except KeyError:
            tz = self._offset_cache[offset] = datetime_timezone(timedelta(minutes=offset))
        return tz.fromutc(dt.replace(tzinfo=tz))

    def _isdst(self, dt: datetime) -> bool:
        tt = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.weekday(), 0, 0)
        stamp = _time.mktime(tt)
        tt = _time.localtime(stamp)
        return tt.tm_isdst > 0