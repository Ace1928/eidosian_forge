import sys
from math import trunc
from typing import (
class EnglishLocale(Locale):
    names = ['en', 'en-us', 'en-gb', 'en-au', 'en-be', 'en-jp', 'en-za', 'en-ca', 'en-ph']
    past = '{0} ago'
    future = 'in {0}'
    and_word = 'and'
    timeframes = {'now': 'just now', 'second': 'a second', 'seconds': '{0} seconds', 'minute': 'a minute', 'minutes': '{0} minutes', 'hour': 'an hour', 'hours': '{0} hours', 'day': 'a day', 'days': '{0} days', 'week': 'a week', 'weeks': '{0} weeks', 'month': 'a month', 'months': '{0} months', 'quarter': 'a quarter', 'quarters': '{0} quarters', 'year': 'a year', 'years': '{0} years'}
    meridians = {'am': 'am', 'pm': 'pm', 'AM': 'AM', 'PM': 'PM'}
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_names = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_abbreviations = ['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ordinal_day_re = '((?P<value>[2-3]?1(?=st)|[2-3]?2(?=nd)|[2-3]?3(?=rd)|[1-3]?[04-9](?=th)|1[1-3](?=th))(st|nd|rd|th))'

    def _ordinal_number(self, n: int) -> str:
        if n % 100 not in (11, 12, 13):
            remainder = abs(n) % 10
            if remainder == 1:
                return f'{n}st'
            elif remainder == 2:
                return f'{n}nd'
            elif remainder == 3:
                return f'{n}rd'
        return f'{n}th'

    def describe(self, timeframe: TimeFrameLiteral, delta: Union[int, float]=0, only_distance: bool=False) -> str:
        """Describes a delta within a timeframe in plain language.

        :param timeframe: a string representing a timeframe.
        :param delta: a quantity representing a delta in a timeframe.
        :param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords
        """
        humanized = super().describe(timeframe, delta, only_distance)
        if only_distance and timeframe == 'now':
            humanized = 'instantly'
        return humanized