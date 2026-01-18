import sys
from math import trunc
from typing import (
class PolishLocale(SlavicBaseLocale):
    names = ['pl', 'pl-pl']
    past = '{0} temu'
    future = 'za {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'teraz', 'second': 'sekundę', 'seconds': {'singular': '{0} sekund', 'dual': '{0} sekundy', 'plural': '{0} sekund'}, 'minute': 'minutę', 'minutes': {'singular': '{0} minut', 'dual': '{0} minuty', 'plural': '{0} minut'}, 'hour': 'godzinę', 'hours': {'singular': '{0} godzin', 'dual': '{0} godziny', 'plural': '{0} godzin'}, 'day': 'dzień', 'days': '{0} dni', 'week': 'tydzień', 'weeks': {'singular': '{0} tygodni', 'dual': '{0} tygodnie', 'plural': '{0} tygodni'}, 'month': 'miesiąc', 'months': {'singular': '{0} miesięcy', 'dual': '{0} miesiące', 'plural': '{0} miesięcy'}, 'year': 'rok', 'years': {'singular': '{0} lat', 'dual': '{0} lata', 'plural': '{0} lat'}}
    month_names = ['', 'styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec', 'sierpień', 'wrzesień', 'październik', 'listopad', 'grudzień']
    month_abbreviations = ['', 'sty', 'lut', 'mar', 'kwi', 'maj', 'cze', 'lip', 'sie', 'wrz', 'paź', 'lis', 'gru']
    day_names = ['', 'poniedziałek', 'wtorek', 'środa', 'czwartek', 'piątek', 'sobota', 'niedziela']
    day_abbreviations = ['', 'Pn', 'Wt', 'Śr', 'Czw', 'Pt', 'So', 'Nd']