import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
class LazyDate:

    @classmethod
    def dtime(cls, dt: str=None, prefer: str='past'):
        if not dt:
            return datetime.now(timezone.utc).isoformat('T')
        pdt = dateparser.parse(dt, settings={'PREFER_DATES_FROM': prefer, 'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
        if prefer == 'past':
            r = datetime.now(timezone.utc) - pdt
        else:
            r = pdt - datetime.now(timezone.utc)
        return TimeItem(r.total_seconds())

    @classmethod
    def date(cls, dt: str=None, prefer: str='past'):
        if not dt:
            return datetime.now(timezone.utc)
        return dateparser.parse(dt, settings={'PREFER_DATES_FROM': prefer, 'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})

    @classmethod
    def fmt(cls, secs, **kwargs):
        return TimeItem(secs)

    @property
    def now(self):
        return self.dtime()

    def __call__(self, **kwargs):
        return self.dtime(**kwargs)