from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
class DateTimeFormat:

    def __init__(self, value: datetime.date | datetime.time, locale: Locale | str, reference_date: datetime.date | None=None) -> None:
        assert isinstance(value, (datetime.date, datetime.datetime, datetime.time))
        if isinstance(value, (datetime.datetime, datetime.time)) and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        self.value = value
        self.locale = Locale.parse(locale)
        self.reference_date = reference_date

    def __getitem__(self, name: str) -> str:
        char = name[0]
        num = len(name)
        if char == 'G':
            return self.format_era(char, num)
        elif char in ('y', 'Y', 'u'):
            return self.format_year(char, num)
        elif char in ('Q', 'q'):
            return self.format_quarter(char, num)
        elif char in ('M', 'L'):
            return self.format_month(char, num)
        elif char in ('w', 'W'):
            return self.format_week(char, num)
        elif char == 'd':
            return self.format(self.value.day, num)
        elif char == 'D':
            return self.format_day_of_year(num)
        elif char == 'F':
            return self.format_day_of_week_in_month()
        elif char in ('E', 'e', 'c'):
            return self.format_weekday(char, num)
        elif char in ('a', 'b', 'B'):
            return self.format_period(char, num)
        elif char == 'h':
            if self.value.hour % 12 == 0:
                return self.format(12, num)
            else:
                return self.format(self.value.hour % 12, num)
        elif char == 'H':
            return self.format(self.value.hour, num)
        elif char == 'K':
            return self.format(self.value.hour % 12, num)
        elif char == 'k':
            if self.value.hour == 0:
                return self.format(24, num)
            else:
                return self.format(self.value.hour, num)
        elif char == 'm':
            return self.format(self.value.minute, num)
        elif char == 's':
            return self.format(self.value.second, num)
        elif char == 'S':
            return self.format_frac_seconds(num)
        elif char == 'A':
            return self.format_milliseconds_in_day(num)
        elif char in ('z', 'Z', 'v', 'V', 'x', 'X', 'O'):
            return self.format_timezone(char, num)
        else:
            raise KeyError(f'Unsupported date/time field {char!r}')

    def extract(self, char: str) -> int:
        char = str(char)[0]
        if char == 'y':
            return self.value.year
        elif char == 'M':
            return self.value.month
        elif char == 'd':
            return self.value.day
        elif char == 'H':
            return self.value.hour
        elif char == 'h':
            return self.value.hour % 12 or 12
        elif char == 'm':
            return self.value.minute
        elif char == 'a':
            return int(self.value.hour >= 12)
        else:
            raise NotImplementedError(f'Not implemented: extracting {char!r} from {self.value!r}')

    def format_era(self, char: str, num: int) -> str:
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[max(3, num)]
        era = int(self.value.year >= 0)
        return get_era_names(width, self.locale)[era]

    def format_year(self, char: str, num: int) -> str:
        value = self.value.year
        if char.isupper():
            value = self.value.isocalendar()[0]
        year = self.format(value, num)
        if num == 2:
            year = year[-2:]
        return year

    def format_quarter(self, char: str, num: int) -> str:
        quarter = (self.value.month - 1) // 3 + 1
        if num <= 2:
            return '%0*d' % (num, quarter)
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[num]
        context = {'Q': 'format', 'q': 'stand-alone'}[char]
        return get_quarter_names(width, context, self.locale)[quarter]

    def format_month(self, char: str, num: int) -> str:
        if num <= 2:
            return '%0*d' % (num, self.value.month)
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[num]
        context = {'M': 'format', 'L': 'stand-alone'}[char]
        return get_month_names(width, context, self.locale)[self.value.month]

    def format_week(self, char: str, num: int) -> str:
        if char.islower():
            day_of_year = self.get_day_of_year()
            week = self.get_week_number(day_of_year)
            if week == 0:
                date = self.value - datetime.timedelta(days=day_of_year)
                week = self.get_week_number(self.get_day_of_year(date), date.weekday())
            return self.format(week, num)
        else:
            week = self.get_week_number(self.value.day)
            if week == 0:
                date = self.value - datetime.timedelta(days=self.value.day)
                week = self.get_week_number(date.day, date.weekday())
            return str(week)

    def format_weekday(self, char: str='E', num: int=4) -> str:
        """
        Return weekday from parsed datetime according to format pattern.

        >>> from datetime import date
        >>> format = DateTimeFormat(date(2016, 2, 28), Locale.parse('en_US'))
        >>> format.format_weekday()
        u'Sunday'

        'E': Day of week - Use one through three letters for the abbreviated day name, four for the full (wide) name,
             five for the narrow name, or six for the short name.
        >>> format.format_weekday('E',2)
        u'Sun'

        'e': Local day of week. Same as E except adds a numeric value that will depend on the local starting day of the
             week, using one or two letters. For this example, Monday is the first day of the week.
        >>> format.format_weekday('e',2)
        '01'

        'c': Stand-Alone local day of week - Use one letter for the local numeric value (same as 'e'), three for the
             abbreviated day name, four for the full (wide) name, five for the narrow name, or six for the short name.
        >>> format.format_weekday('c',1)
        '1'

        :param char: pattern format character ('e','E','c')
        :param num: count of format character

        """
        if num < 3:
            if char.islower():
                value = 7 - self.locale.first_week_day + self.value.weekday()
                return self.format(value % 7 + 1, num)
            num = 3
        weekday = self.value.weekday()
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow', 6: 'short'}[num]
        context = 'stand-alone' if char == 'c' else 'format'
        return get_day_names(width, context, self.locale)[weekday]

    def format_day_of_year(self, num: int) -> str:
        return self.format(self.get_day_of_year(), num)

    def format_day_of_week_in_month(self) -> str:
        return str((self.value.day - 1) // 7 + 1)

    def format_period(self, char: str, num: int) -> str:
        """
        Return period from parsed datetime according to format pattern.

        >>> from datetime import datetime, time
        >>> format = DateTimeFormat(time(13, 42), 'fi_FI')
        >>> format.format_period('a', 1)
        u'ip.'
        >>> format.format_period('b', 1)
        u'iltap.'
        >>> format.format_period('b', 4)
        u'iltapäivä'
        >>> format.format_period('B', 4)
        u'iltapäivällä'
        >>> format.format_period('B', 5)
        u'ip.'

        >>> format = DateTimeFormat(datetime(2022, 4, 28, 6, 27), 'zh_Hant')
        >>> format.format_period('a', 1)
        u'上午'
        >>> format.format_period('b', 1)
        u'清晨'
        >>> format.format_period('B', 1)
        u'清晨'

        :param char: pattern format character ('a', 'b', 'B')
        :param num: count of format character

        """
        widths = [{3: 'abbreviated', 4: 'wide', 5: 'narrow'}[max(3, num)], 'wide', 'narrow', 'abbreviated']
        if char == 'a':
            period = 'pm' if self.value.hour >= 12 else 'am'
            context = 'format'
        else:
            period = get_period_id(self.value, locale=self.locale)
            context = 'format' if char == 'B' else 'stand-alone'
        for width in widths:
            period_names = get_period_names(context=context, width=width, locale=self.locale)
            if period in period_names:
                return period_names[period]
        raise ValueError(f'Could not format period {period} in {self.locale}')

    def format_frac_seconds(self, num: int) -> str:
        """ Return fractional seconds.

        Rounds the time's microseconds to the precision given by the number         of digits passed in.
        """
        value = self.value.microsecond / 1000000
        return self.format(round(value, num) * 10 ** num, num)

    def format_milliseconds_in_day(self, num):
        msecs = self.value.microsecond // 1000 + self.value.second * 1000 + self.value.minute * 60000 + self.value.hour * 3600000
        return self.format(msecs, num)

    def format_timezone(self, char: str, num: int) -> str:
        width = {3: 'short', 4: 'long', 5: 'iso8601'}[max(3, num)]
        value = self.value
        if self.reference_date:
            value = datetime.datetime.combine(self.reference_date, self.value)
        if char == 'z':
            return get_timezone_name(value, width, locale=self.locale)
        elif char == 'Z':
            if num == 5:
                return get_timezone_gmt(value, width, locale=self.locale, return_z=True)
            return get_timezone_gmt(value, width, locale=self.locale)
        elif char == 'O':
            if num == 4:
                return get_timezone_gmt(value, width, locale=self.locale)
        elif char == 'v':
            return get_timezone_name(value.tzinfo, width, locale=self.locale)
        elif char == 'V':
            if num == 1:
                return get_timezone_name(value.tzinfo, width, uncommon=True, locale=self.locale)
            elif num == 2:
                return get_timezone_name(value.tzinfo, locale=self.locale, return_zone=True)
            elif num == 3:
                return get_timezone_location(value.tzinfo, locale=self.locale, return_city=True)
            return get_timezone_location(value.tzinfo, locale=self.locale)
        elif char == 'X':
            if num == 1:
                return get_timezone_gmt(value, width='iso8601_short', locale=self.locale, return_z=True)
            elif num in (2, 4):
                return get_timezone_gmt(value, width='short', locale=self.locale, return_z=True)
            elif num in (3, 5):
                return get_timezone_gmt(value, width='iso8601', locale=self.locale, return_z=True)
        elif char == 'x':
            if num == 1:
                return get_timezone_gmt(value, width='iso8601_short', locale=self.locale)
            elif num in (2, 4):
                return get_timezone_gmt(value, width='short', locale=self.locale)
            elif num in (3, 5):
                return get_timezone_gmt(value, width='iso8601', locale=self.locale)

    def format(self, value: SupportsInt, length: int) -> str:
        return '%0*d' % (length, value)

    def get_day_of_year(self, date: datetime.date | None=None) -> int:
        if date is None:
            date = self.value
        return (date - date.replace(month=1, day=1)).days + 1

    def get_week_number(self, day_of_period: int, day_of_week: int | None=None) -> int:
        """Return the number of the week of a day within a period. This may be
        the week number in a year or the week number in a month.

        Usually this will return a value equal to or greater than 1, but if the
        first week of the period is so short that it actually counts as the last
        week of the previous period, this function will return 0.

        >>> date = datetime.date(2006, 1, 8)
        >>> DateTimeFormat(date, 'de_DE').get_week_number(6)
        1
        >>> DateTimeFormat(date, 'en_US').get_week_number(6)
        2

        :param day_of_period: the number of the day in the period (usually
                              either the day of month or the day of year)
        :param day_of_week: the week day; if omitted, the week day of the
                            current date is assumed
        """
        if day_of_week is None:
            day_of_week = self.value.weekday()
        first_day = (day_of_week - self.locale.first_week_day - day_of_period + 1) % 7
        if first_day < 0:
            first_day += 7
        week_number = (day_of_period + first_day - 1) // 7
        if 7 - first_day >= self.locale.min_week_days:
            week_number += 1
        if self.locale.first_week_day == 0:
            max_weeks = datetime.date(year=self.value.year, day=28, month=12).isocalendar()[1]
            if week_number > max_weeks:
                week_number -= max_weeks
        return week_number