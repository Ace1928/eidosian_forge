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
def _viable_freqs(min_breaks: int, unit_durations: tuple[int, int, int, int, int, int, int]) -> Generator[tuple[DateFrequency, int], None, None]:
    """
    Find viable frequency, duration pairs

    A pair is viable if it can yeild a suitable number of breaks
    For example:
        - YEARLY frequency, 3 year unit_duration and
          8 min_breaks is not viable
        - MONTHLY frequency, 36 month unit_duration and
          8 min_breaks is viable
    """
    for freq, duration in zip(DateFrequency, unit_durations):
        max_width = WIDTHS[freq][-1]
        max_breaks = max(min_breaks, MAX_BREAKS.get(freq, 11))
        if duration <= max_width * max_breaks - 1:
            yield (freq, duration)