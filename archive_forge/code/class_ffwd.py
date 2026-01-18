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
class ffwd:
    """Version of ``dateutil.relativedelta`` that only supports addition."""

    def __init__(self, year=None, month=None, weeks=0, weekday=None, day=None, hour=None, minute=None, second=None, microsecond=None, **kwargs: Any):
        self.year = year
        self.month = month
        self.weeks = weeks
        self.weekday = weekday
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.days = weeks * 7
        self._has_time = self.hour is not None or self.minute is not None

    def __repr__(self) -> str:
        return reprcall('ffwd', (), self._fields(weeks=self.weeks, weekday=self.weekday))

    def __radd__(self, other: Any) -> timedelta:
        if not isinstance(other, date):
            return NotImplemented
        year = self.year or other.year
        month = self.month or other.month
        day = min(monthrange(year, month)[1], self.day or other.day)
        ret = other.replace(**dict(dictfilter(self._fields()), year=year, month=month, day=day))
        if self.weekday is not None:
            ret += timedelta(days=(7 - ret.weekday() + self.weekday) % 7)
        return ret + timedelta(days=self.days)

    def _fields(self, **extra: Any) -> dict[str, Any]:
        return dictfilter({'year': self.year, 'month': self.month, 'day': self.day, 'hour': self.hour, 'minute': self.minute, 'second': self.second, 'microsecond': self.microsecond}, **extra)