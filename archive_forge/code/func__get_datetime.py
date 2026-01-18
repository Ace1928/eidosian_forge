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
def _get_datetime(cls, expr: Union['Arrow', dt_datetime, int, float, str]) -> dt_datetime:
    """Get datetime object from a specified expression."""
    if isinstance(expr, Arrow):
        return expr.datetime
    elif isinstance(expr, dt_datetime):
        return expr
    elif util.is_timestamp(expr):
        timestamp = float(expr)
        return cls.utcfromtimestamp(timestamp).datetime
    else:
        raise ValueError(f'{expr!r} not recognized as a datetime or timestamp.')