import calendar
import re
import sys
from datetime import date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta
from datetime import tzinfo as dt_tzinfo
from math import trunc
from time import struct_time
from typing import (
from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta
from arrow import formatter, locales, parser, util
from arrow.constants import DEFAULT_LOCALE, DEHUMANIZE_LOCALES
from arrow.locales import TimeFrameLiteral
@classmethod
def _get_iteration_params(cls, end: Any, limit: Optional[int]) -> Tuple[Any, int]:
    """Sets default end and limit values for range method."""
    if end is None:
        if limit is None:
            raise ValueError("One of 'end' or 'limit' is required.")
        return (cls.max, limit)
    else:
        if limit is None:
            return (end, sys.maxsize)
        return (end, limit)