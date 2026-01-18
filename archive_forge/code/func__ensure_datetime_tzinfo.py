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
def _ensure_datetime_tzinfo(dt: datetime.datetime, tzinfo: datetime.tzinfo | None=None) -> datetime.datetime:
    """
    Ensure the datetime passed has an attached tzinfo.

    If the datetime is tz-naive to begin with, UTC is attached.

    If a tzinfo is passed in, the datetime is normalized to that timezone.

    >>> from datetime import datetime
    >>> _get_tz_name(_ensure_datetime_tzinfo(datetime(2015, 1, 1)))
    'UTC'

    >>> tz = get_timezone("Europe/Stockholm")
    >>> _ensure_datetime_tzinfo(datetime(2015, 1, 1, 13, 15, tzinfo=UTC), tzinfo=tz).hour
    14

    :param datetime: Datetime to augment.
    :param tzinfo: optional tzinfo
    :return: datetime with tzinfo
    :rtype: datetime
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if tzinfo is not None:
        dt = dt.astimezone(get_timezone(tzinfo))
        if hasattr(tzinfo, 'normalize'):
            dt = tzinfo.normalize(dt)
    return dt