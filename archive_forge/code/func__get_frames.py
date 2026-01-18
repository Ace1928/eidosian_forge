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
def _get_frames(cls, name: _T_FRAMES) -> Tuple[str, str, int]:
    """Finds relevant timeframe and steps for use in range and span methods.

        Returns a 3 element tuple in the form (frame, plural frame, step), for example ("day", "days", 1)

        """
    if name in cls._ATTRS:
        return (name, f'{name}s', 1)
    elif name[-1] == 's' and name[:-1] in cls._ATTRS:
        return (name[:-1], name, 1)
    elif name in ['week', 'weeks']:
        return ('week', 'weeks', 1)
    elif name in ['quarter', 'quarters']:
        return ('quarter', 'months', 3)
    else:
        supported = ', '.join(['year(s)', 'month(s)', 'day(s)', 'hour(s)', 'minute(s)', 'second(s)', 'microsecond(s)', 'week(s)', 'quarter(s)'])
        raise ValueError(f'Range or span over frame {name} not supported. Supported frames: {supported}.')