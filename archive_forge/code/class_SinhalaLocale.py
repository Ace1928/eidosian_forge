import sys
from math import trunc
from typing import (
class SinhalaLocale(Locale):
    names = ['si', 'si-lk']
    past = '{0}ට පෙර'
    future = '{0}'
    and_word = 'සහ'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[Mapping[str, str], str]]] = {'now': 'දැන්', 'second': {'past': 'තත්පරයක', 'future': 'තත්පරයකින්'}, 'seconds': {'past': 'තත්පර {0} ක', 'future': 'තත්පර {0} කින්'}, 'minute': {'past': 'විනාඩියක', 'future': 'විනාඩියකින්'}, 'minutes': {'past': 'විනාඩි {0} ක', 'future': 'මිනිත්තු {0} කින්'}, 'hour': {'past': 'පැයක', 'future': 'පැයකින්'}, 'hours': {'past': 'පැය {0} ක', 'future': 'පැය {0} කින්'}, 'day': {'past': 'දිනක', 'future': 'දිනකට'}, 'days': {'past': 'දින {0} ක', 'future': 'දින {0} කින්'}, 'week': {'past': 'සතියක', 'future': 'සතියකින්'}, 'weeks': {'past': 'සති {0} ක', 'future': 'සති {0} කින්'}, 'month': {'past': 'මාසයක', 'future': 'එය මාසය තුළ'}, 'months': {'past': 'මාස {0} ක', 'future': 'මාස {0} කින්'}, 'year': {'past': 'වසරක', 'future': 'වසරක් තුළ'}, 'years': {'past': 'අවුරුදු {0} ක', 'future': 'අවුරුදු {0} තුළ'}}
    timeframes_only_distance = {}
    timeframes_only_distance['second'] = 'තත්පරයක්'
    timeframes_only_distance['seconds'] = 'තත්පර {0}'
    timeframes_only_distance['minute'] = 'මිනිත්තුවක්'
    timeframes_only_distance['minutes'] = 'විනාඩි {0}'
    timeframes_only_distance['hour'] = 'පැයක්'
    timeframes_only_distance['hours'] = 'පැය {0}'
    timeframes_only_distance['day'] = 'දවසක්'
    timeframes_only_distance['days'] = 'දවස් {0}'
    timeframes_only_distance['week'] = 'සතියක්'
    timeframes_only_distance['weeks'] = 'සති {0}'
    timeframes_only_distance['month'] = 'මාසයක්'
    timeframes_only_distance['months'] = 'මාස {0}'
    timeframes_only_distance['year'] = 'අවුරුද්දක්'
    timeframes_only_distance['years'] = 'අවුරුදු {0}'

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        """
        Sinhala awares time frame format function, takes into account
        the differences between general, past, and future forms (three different suffixes).
        """
        abs_delta = abs(delta)
        form = self.timeframes[timeframe]
        if isinstance(form, str):
            return form.format(abs_delta)
        if delta > 0:
            key = 'future'
        else:
            key = 'past'
        form = form[key]
        return form.format(abs_delta)

    def describe(self, timeframe: TimeFrameLiteral, delta: Union[float, int]=1, only_distance: bool=False) -> str:
        """Describes a delta within a timeframe in plain language.

        :param timeframe: a string representing a timeframe.
        :param delta: a quantity representing a delta in a timeframe.
        :param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords
        """
        if not only_distance:
            return super().describe(timeframe, delta, only_distance)
        humanized = self.timeframes_only_distance[timeframe].format(trunc(abs(delta)))
        return humanized
    month_names = ['', 'ජනවාරි', 'පෙබරවාරි', 'මාර්තු', 'අප්\u200dරේල්', 'මැයි', 'ජූනි', 'ජූලි', 'අගෝස්තු', 'සැප්තැම්බර්', 'ඔක්තෝබර්', 'නොවැම්බර්', 'දෙසැම්බර්']
    month_abbreviations = ['', 'ජන', 'පෙබ', 'මාර්', 'අප්\u200dරේ', 'මැයි', 'ජුනි', 'ජූලි', 'අගෝ', 'සැප්', 'ඔක්', 'නොවැ', 'දෙසැ']
    day_names = ['', 'සදුදා', 'අඟහරැවදා', 'බදාදා', 'බ්\u200dරහස්\u200dපතින්\u200dදා', 'සිකුරාදා', 'සෙනසුරාදා', 'ඉරිදා']
    day_abbreviations = ['', 'සදුද', 'බදා', 'බදා', 'සිකු', 'සෙන', 'අ', 'ඉරිදා']