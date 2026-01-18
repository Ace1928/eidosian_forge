import sys
from math import trunc
from typing import (
class SerbianLocale(Locale):
    names = ['sr', 'sr-rs', 'sr-sp']
    past = 'pre {0}'
    future = 'za {0}'
    and_word = 'i'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'sada', 'second': 'sekundu', 'seconds': {'double': '{0} sekunde', 'higher': '{0} sekundi'}, 'minute': 'minutu', 'minutes': {'double': '{0} minute', 'higher': '{0} minuta'}, 'hour': 'sat', 'hours': {'double': '{0} sata', 'higher': '{0} sati'}, 'day': 'dan', 'days': {'double': '{0} dana', 'higher': '{0} dana'}, 'week': 'nedelju', 'weeks': {'double': '{0} nedelje', 'higher': '{0} nedelja'}, 'month': 'mesec', 'months': {'double': '{0} meseca', 'higher': '{0} meseci'}, 'year': 'godinu', 'years': {'double': '{0} godine', 'higher': '{0} godina'}}
    month_names = ['', 'januar', 'februar', 'mart', 'april', 'maj', 'jun', 'jul', 'avgust', 'septembar', 'oktobar', 'novembar', 'decembar']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'maj', 'jun', 'jul', 'avg', 'sep', 'okt', 'nov', 'dec']
    day_names = ['', 'ponedeljak', 'utorak', 'sreda', 'četvrtak', 'petak', 'subota', 'nedelja']
    day_abbreviations = ['', 'po', 'ut', 'sr', 'če', 'pe', 'su', 'ne']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        delta = abs(delta)
        if isinstance(form, Mapping):
            if 1 < delta <= 4:
                form = form['double']
            else:
                form = form['higher']
        return form.format(delta)