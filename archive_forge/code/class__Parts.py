import re
import sys
from datetime import datetime, timedelta
from datetime import tzinfo as dt_tzinfo
from functools import lru_cache
from typing import (
from dateutil import tz
from arrow import locales
from arrow.constants import DEFAULT_LOCALE
from arrow.util import next_weekday, normalize_timestamp
class _Parts(TypedDict, total=False):
    year: int
    month: int
    day_of_year: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    timestamp: float
    expanded_timestamp: int
    tzinfo: dt_tzinfo
    am_pm: Literal['am', 'pm']
    day_of_week: int
    weekdate: Tuple[_WEEKDATE_ELEMENT, _WEEKDATE_ELEMENT, Optional[_WEEKDATE_ELEMENT]]