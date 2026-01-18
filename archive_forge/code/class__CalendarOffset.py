import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
class _CalendarOffset:
    __slots__ = ['m', 'w', 'd', 'hour', 'minute', 'second']
    _DAYS_BEFORE_MONTH = (-1, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)

    def __init__(self, m, w, d, hour=2, minute=0, second=0):
        if not 1 <= m <= 12:
            raise ValueError('m must be in [1, 12]')
        if not 1 <= w <= 5:
            raise ValueError('w must be in [1, 5]')
        if not 0 <= d <= 6:
            raise ValueError('d must be in [0, 6]')
        self.m = m
        self.w = w
        self.d = d
        self.hour = hour
        self.minute = minute
        self.second = second

    @classmethod
    def _ymd2ord(cls, year, month, day):
        return _post_epoch_days_before_year(year) + cls._DAYS_BEFORE_MONTH[month] + (month > 2 and calendar.isleap(year)) + day

    def year_to_epoch(self, year):
        """Calculates the datetime of the occurrence from the year"""
        first_day, days_in_month = calendar.monthrange(year, self.m)
        month_day = (self.d - (first_day + 1)) % 7 + 1
        month_day += (self.w - 1) * 7
        if month_day > days_in_month:
            month_day -= 7
        ordinal = self._ymd2ord(year, self.m, month_day)
        epoch = ordinal * 86400
        epoch += self.hour * 3600 + self.minute * 60 + self.second
        return epoch