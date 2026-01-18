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
def _parse_token(self, token: Any, value: Any, parts: _Parts) -> None:
    if token == 'YYYY':
        parts['year'] = int(value)
    elif token == 'YY':
        value = int(value)
        parts['year'] = 1900 + value if value > 68 else 2000 + value
    elif token in ['MMMM', 'MMM']:
        parts['month'] = self.locale.month_number(value.lower())
    elif token in ['MM', 'M']:
        parts['month'] = int(value)
    elif token in ['DDDD', 'DDD']:
        parts['day_of_year'] = int(value)
    elif token in ['DD', 'D']:
        parts['day'] = int(value)
    elif token == 'Do':
        parts['day'] = int(value)
    elif token == 'dddd':
        day_of_week = [x.lower() for x in self.locale.day_names].index(value.lower())
        parts['day_of_week'] = day_of_week - 1
    elif token == 'ddd':
        day_of_week = [x.lower() for x in self.locale.day_abbreviations].index(value.lower())
        parts['day_of_week'] = day_of_week - 1
    elif token.upper() in ['HH', 'H']:
        parts['hour'] = int(value)
    elif token in ['mm', 'm']:
        parts['minute'] = int(value)
    elif token in ['ss', 's']:
        parts['second'] = int(value)
    elif token == 'S':
        value = value.ljust(7, '0')
        seventh_digit = int(value[6])
        if seventh_digit == 5:
            rounding = int(value[5]) % 2
        elif seventh_digit > 5:
            rounding = 1
        else:
            rounding = 0
        parts['microsecond'] = int(value[:6]) + rounding
    elif token == 'X':
        parts['timestamp'] = float(value)
    elif token == 'x':
        parts['expanded_timestamp'] = int(value)
    elif token in ['ZZZ', 'ZZ', 'Z']:
        parts['tzinfo'] = TzinfoParser.parse(value)
    elif token in ['a', 'A']:
        if value in (self.locale.meridians['am'], self.locale.meridians['AM']):
            parts['am_pm'] = 'am'
            if 'hour' in parts and (not 0 <= parts['hour'] <= 12):
                raise ParserMatchError(f'Hour token value must be between 0 and 12 inclusive for token {token!r}.')
        elif value in (self.locale.meridians['pm'], self.locale.meridians['PM']):
            parts['am_pm'] = 'pm'
    elif token == 'W':
        parts['weekdate'] = value