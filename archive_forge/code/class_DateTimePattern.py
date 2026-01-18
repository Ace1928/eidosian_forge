from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
class DateTimePattern:

    def __init__(self, pattern: str, format: DateTimeFormat):
        self.pattern = pattern
        self.format = format

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.pattern!r}>'

    def __str__(self) -> str:
        pat = self.pattern
        return pat

    def __mod__(self, other: DateTimeFormat) -> str:
        if not isinstance(other, DateTimeFormat):
            return NotImplemented
        return self.format % other

    def apply(self, datetime: datetime.date | datetime.time, locale: Locale | str | None, reference_date: datetime.date | None=None) -> str:
        return self % DateTimeFormat(datetime, locale, reference_date)