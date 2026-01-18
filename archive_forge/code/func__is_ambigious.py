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
def _is_ambigious(dt: datetime, tz: tzinfo) -> bool:
    """Helper function to determine if a timezone is ambiguous using python's dateutil module.

    Returns False if the timezone cannot detect ambiguity, or if there is no ambiguity, otherwise True.

    In order to detect ambiguous datetimes, the timezone must be built using ZoneInfo, or have an is_ambiguous
    method. Previously, pytz timezones would throw an AmbiguousTimeError if the localized dt was ambiguous,
    but now we need to specifically check for ambiguity with dateutil, as pytz is deprecated.
    """
    return _can_detect_ambiguous(tz) and dateutil_tz.datetime_ambiguous(dt)