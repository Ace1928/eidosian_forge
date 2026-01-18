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
def _get_tz_name(dt_or_tzinfo: _DtOrTzinfo) -> str:
    """
    Get the timezone name out of a time, datetime, or tzinfo object.

    :rtype: str
    """
    dt, tzinfo = _get_dt_and_tzinfo(dt_or_tzinfo)
    if hasattr(tzinfo, 'zone'):
        return tzinfo.zone
    elif hasattr(tzinfo, 'key') and tzinfo.key is not None:
        return tzinfo.key
    else:
        return tzinfo.tzname(dt or datetime.datetime.now(UTC))