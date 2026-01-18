import sys
from math import trunc
from typing import (
class FinnishLocale(Locale):
    names = ['fi', 'fi-fi']
    past = '{0} sitten'
    future = '{0} kuluttua'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'juuri nyt', 'second': {'past': 'sekunti', 'future': 'sekunnin'}, 'seconds': {'past': '{0} sekuntia', 'future': '{0} sekunnin'}, 'minute': {'past': 'minuutti', 'future': 'minuutin'}, 'minutes': {'past': '{0} minuuttia', 'future': '{0} minuutin'}, 'hour': {'past': 'tunti', 'future': 'tunnin'}, 'hours': {'past': '{0} tuntia', 'future': '{0} tunnin'}, 'day': {'past': 'päivä', 'future': 'päivän'}, 'days': {'past': '{0} päivää', 'future': '{0} päivän'}, 'week': {'past': 'viikko', 'future': 'viikon'}, 'weeks': {'past': '{0} viikkoa', 'future': '{0} viikon'}, 'month': {'past': 'kuukausi', 'future': 'kuukauden'}, 'months': {'past': '{0} kuukautta', 'future': '{0} kuukauden'}, 'year': {'past': 'vuosi', 'future': 'vuoden'}, 'years': {'past': '{0} vuotta', 'future': '{0} vuoden'}}
    month_names = ['', 'tammikuu', 'helmikuu', 'maaliskuu', 'huhtikuu', 'toukokuu', 'kesäkuu', 'heinäkuu', 'elokuu', 'syyskuu', 'lokakuu', 'marraskuu', 'joulukuu']
    month_abbreviations = ['', 'tammi', 'helmi', 'maalis', 'huhti', 'touko', 'kesä', 'heinä', 'elo', 'syys', 'loka', 'marras', 'joulu']
    day_names = ['', 'maanantai', 'tiistai', 'keskiviikko', 'torstai', 'perjantai', 'lauantai', 'sunnuntai']
    day_abbreviations = ['', 'ma', 'ti', 'ke', 'to', 'pe', 'la', 'su']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        if isinstance(form, Mapping):
            if delta < 0:
                form = form['past']
            else:
                form = form['future']
        return form.format(abs(delta))

    def _ordinal_number(self, n: int) -> str:
        return f'{n}.'