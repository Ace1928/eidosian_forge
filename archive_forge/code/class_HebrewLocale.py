import sys
from math import trunc
from typing import (
class HebrewLocale(Locale):
    names = ['he', 'he-il']
    past = 'לפני {0}'
    future = 'בעוד {0}'
    and_word = 'ו'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'הרגע', 'second': 'שנייה', 'seconds': '{0} שניות', 'minute': 'דקה', 'minutes': '{0} דקות', 'hour': 'שעה', 'hours': {'2': 'שעתיים', 'ten': '{0} שעות', 'higher': '{0} שעות'}, 'day': 'יום', 'days': {'2': 'יומיים', 'ten': '{0} ימים', 'higher': '{0} יום'}, 'week': 'שבוע', 'weeks': {'2': 'שבועיים', 'ten': '{0} שבועות', 'higher': '{0} שבועות'}, 'month': 'חודש', 'months': {'2': 'חודשיים', 'ten': '{0} חודשים', 'higher': '{0} חודשים'}, 'year': 'שנה', 'years': {'2': 'שנתיים', 'ten': '{0} שנים', 'higher': '{0} שנה'}}
    meridians = {'am': 'לפנ"צ', 'pm': 'אחר"צ', 'AM': 'לפני הצהריים', 'PM': 'אחרי הצהריים'}
    month_names = ['', 'ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר']
    month_abbreviations = ['', 'ינו׳', 'פבר׳', 'מרץ', 'אפר׳', 'מאי', 'יוני', 'יולי', 'אוג׳', 'ספט׳', 'אוק׳', 'נוב׳', 'דצמ׳']
    day_names = ['', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת', 'ראשון']
    day_abbreviations = ['', 'ב׳', 'ג׳', 'ד׳', 'ה׳', 'ו׳', 'ש׳', 'א׳']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        delta = abs(delta)
        if isinstance(form, Mapping):
            if delta == 2:
                form = form['2']
            elif delta == 0 or 2 < delta <= 10:
                form = form['ten']
            else:
                form = form['higher']
        return form.format(delta)

    def describe_multi(self, timeframes: Sequence[Tuple[TimeFrameLiteral, Union[int, float]]], only_distance: bool=False) -> str:
        """Describes a delta within multiple timeframes in plain language.
        In Hebrew, the and word behaves a bit differently.

        :param timeframes: a list of string, quantity pairs each representing a timeframe and delta.
        :param only_distance: return only distance eg: "2 hours and 11 seconds" without "in" or "ago" keywords
        """
        humanized = ''
        for index, (timeframe, delta) in enumerate(timeframes):
            last_humanized = self._format_timeframe(timeframe, trunc(delta))
            if index == 0:
                humanized = last_humanized
            elif index == len(timeframes) - 1:
                humanized += ' ' + self.and_word
                if last_humanized[0].isdecimal():
                    humanized += '־'
                humanized += last_humanized
            else:
                humanized += ', ' + last_humanized
        if not only_distance:
            humanized = self._format_relative(humanized, timeframe, trunc(delta))
        return humanized