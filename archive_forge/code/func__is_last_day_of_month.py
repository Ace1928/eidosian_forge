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
@staticmethod
def _is_last_day_of_month(date: 'Arrow') -> bool:
    """Returns a boolean indicating whether the datetime is the last day of the month."""
    return date.day == calendar.monthrange(date.year, date.month)[1]