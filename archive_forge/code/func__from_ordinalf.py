from __future__ import annotations
import math
import typing
from datetime import datetime, timedelta, tzinfo
from typing import overload
from zoneinfo import ZoneInfo
import numpy as np
from dateutil.rrule import rrule
from ..utils import get_timezone, isclose_abs
from .date_utils import Interval, align_limits, expand_datetime_limits
from .types import DateFrequency, date_breaks_info
def _from_ordinalf(x: float, tz: tzinfo | None) -> datetime:
    """
    Convert float array to datetime
    """
    dt64 = EPOCH64 + np.timedelta64(int(np.round(x * MICROSECONDS_PER_DAY)), 'us')
    if not MIN_DATETIME64 < dt64 <= MAX_DATETIME64:
        raise ValueError(f'Date ordinal {x} converts to {dt64} (using epoch {EPOCH}). The supported dates must be  between year 0001 and 9999.')
    dt: datetime = dt64.astype(object)
    if tz:
        dt = dt.replace(tzinfo=UTC)
        dt = dt.astimezone(tz)
    if np.abs(x) > 70 * 365:
        ms = round(dt.microsecond / 20) * 20
        if ms == 1000000:
            dt = dt.replace(microsecond=0) + timedelta(seconds=1)
        else:
            dt = dt.replace(microsecond=ms)
    return dt