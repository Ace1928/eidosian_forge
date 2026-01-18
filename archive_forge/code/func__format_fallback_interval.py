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
def _format_fallback_interval(start: _Instant, end: _Instant, skeleton: str | None, tzinfo: datetime.tzinfo | None, locale: Locale | str | None=LC_TIME) -> str:
    if skeleton in locale.datetime_skeletons:
        format = lambda dt: format_skeleton(skeleton, dt, tzinfo, locale=locale)
    elif all((isinstance(d, datetime.date) and (not isinstance(d, datetime.datetime)) for d in (start, end))):
        format = lambda dt: format_date(dt, locale=locale)
    elif all((isinstance(d, datetime.time) and (not isinstance(d, datetime.date)) for d in (start, end))):
        format = lambda dt: format_time(dt, tzinfo=tzinfo, locale=locale)
    else:
        format = lambda dt: format_datetime(dt, tzinfo=tzinfo, locale=locale)
    formatted_start = format(start)
    formatted_end = format(end)
    if formatted_start == formatted_end:
        return format(start)
    return locale.interval_formats.get(None, '{0}-{1}').replace('{0}', formatted_start).replace('{1}', formatted_end)