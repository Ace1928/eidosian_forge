import sys
from math import trunc
from typing import (
class GermanBaseLocale(Locale):
    past = 'vor {0}'
    future = 'in {0}'
    and_word = 'und'
    timeframes: ClassVar[Dict[TimeFrameLiteral, str]] = {'now': 'gerade eben', 'second': 'einer Sekunde', 'seconds': '{0} Sekunden', 'minute': 'einer Minute', 'minutes': '{0} Minuten', 'hour': 'einer Stunde', 'hours': '{0} Stunden', 'day': 'einem Tag', 'days': '{0} Tagen', 'week': 'einer Woche', 'weeks': '{0} Wochen', 'month': 'einem Monat', 'months': '{0} Monaten', 'year': 'einem Jahr', 'years': '{0} Jahren'}
    timeframes_only_distance = timeframes.copy()
    timeframes_only_distance['second'] = 'eine Sekunde'
    timeframes_only_distance['minute'] = 'eine Minute'
    timeframes_only_distance['hour'] = 'eine Stunde'
    timeframes_only_distance['day'] = 'ein Tag'
    timeframes_only_distance['days'] = '{0} Tage'
    timeframes_only_distance['week'] = 'eine Woche'
    timeframes_only_distance['month'] = 'ein Monat'
    timeframes_only_distance['months'] = '{0} Monate'
    timeframes_only_distance['year'] = 'ein Jahr'
    timeframes_only_distance['years'] = '{0} Jahre'
    month_names = ['', 'Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']
    month_abbreviations = ['', 'Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    day_names = ['', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    day_abbreviations = ['', 'Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

    def _ordinal_number(self, n: int) -> str:
        return f'{n}.'

    def describe(self, timeframe: TimeFrameLiteral, delta: Union[int, float]=0, only_distance: bool=False) -> str:
        """Describes a delta within a timeframe in plain language.

        :param timeframe: a string representing a timeframe.
        :param delta: a quantity representing a delta in a timeframe.
        :param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords
        """
        if not only_distance:
            return super().describe(timeframe, delta, only_distance)
        humanized: str = self.timeframes_only_distance[timeframe].format(trunc(abs(delta)))
        return humanized