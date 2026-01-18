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
class _Zone:
    """Timezone class that provides the timezone for the application.
    If `enable_utc` is disabled, LocalTimezone is provided as the timezone provider through local().
    Otherwise, this class provides a UTC ZoneInfo instance as the timezone provider for the application.

    Additionally this class provides a few utility methods for converting datetimes.
    """

    def tz_or_local(self, tzinfo: tzinfo | None=None) -> tzinfo:
        """Return either our local timezone or the provided timezone."""
        if tzinfo is None:
            return self.local
        return self.get_timezone(tzinfo)

    def to_local(self, dt: datetime, local=None, orig=None):
        """Converts a datetime to the local timezone."""
        if is_naive(dt):
            dt = make_aware(dt, orig or self.utc)
        return localize(dt, self.tz_or_local(local))

    def to_system(self, dt: datetime) -> datetime:
        """Converts a datetime to the system timezone."""
        return dt.astimezone(tz=None)

    def to_local_fallback(self, dt: datetime) -> datetime:
        """Converts a datetime to the local timezone, or the system timezone."""
        if is_naive(dt):
            return make_aware(dt, self.local)
        return localize(dt, self.local)

    def get_timezone(self, zone: str | tzinfo) -> tzinfo:
        """Returns ZoneInfo timezone if the provided zone is a string, otherwise return the zone."""
        if isinstance(zone, str):
            return ZoneInfo(zone)
        return zone

    @cached_property
    def local(self) -> LocalTimezone:
        """Return LocalTimezone instance for the application."""
        return LocalTimezone()

    @cached_property
    def utc(self) -> tzinfo:
        """Return UTC timezone created with ZoneInfo."""
        return self.get_timezone('UTC')