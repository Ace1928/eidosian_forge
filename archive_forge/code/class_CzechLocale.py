import sys
from math import trunc
from typing import (
class CzechLocale(Locale):
    names = ['cs', 'cs-cz']
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'Teď', 'second': {'past': 'vteřina', 'future': 'vteřina'}, 'seconds': {'zero': 'vteřina', 'past': '{0} sekundami', 'future-singular': '{0} sekundy', 'future-paucal': '{0} sekund'}, 'minute': {'past': 'minutou', 'future': 'minutu'}, 'minutes': {'zero': '{0} minut', 'past': '{0} minutami', 'future-singular': '{0} minuty', 'future-paucal': '{0} minut'}, 'hour': {'past': 'hodinou', 'future': 'hodinu'}, 'hours': {'zero': '{0} hodin', 'past': '{0} hodinami', 'future-singular': '{0} hodiny', 'future-paucal': '{0} hodin'}, 'day': {'past': 'dnem', 'future': 'den'}, 'days': {'zero': '{0} dnů', 'past': '{0} dny', 'future-singular': '{0} dny', 'future-paucal': '{0} dnů'}, 'week': {'past': 'týdnem', 'future': 'týden'}, 'weeks': {'zero': '{0} týdnů', 'past': '{0} týdny', 'future-singular': '{0} týdny', 'future-paucal': '{0} týdnů'}, 'month': {'past': 'měsícem', 'future': 'měsíc'}, 'months': {'zero': '{0} měsíců', 'past': '{0} měsíci', 'future-singular': '{0} měsíce', 'future-paucal': '{0} měsíců'}, 'year': {'past': 'rokem', 'future': 'rok'}, 'years': {'zero': '{0} let', 'past': '{0} lety', 'future-singular': '{0} roky', 'future-paucal': '{0} let'}}
    past = 'Před {0}'
    future = 'Za {0}'
    month_names = ['', 'leden', 'únor', 'březen', 'duben', 'květen', 'červen', 'červenec', 'srpen', 'září', 'říjen', 'listopad', 'prosinec']
    month_abbreviations = ['', 'led', 'úno', 'bře', 'dub', 'kvě', 'čvn', 'čvc', 'srp', 'zář', 'říj', 'lis', 'pro']
    day_names = ['', 'pondělí', 'úterý', 'středa', 'čtvrtek', 'pátek', 'sobota', 'neděle']
    day_abbreviations = ['', 'po', 'út', 'st', 'čt', 'pá', 'so', 'ne']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        """Czech aware time frame format function, takes into account
        the differences between past and future forms."""
        abs_delta = abs(delta)
        form = self.timeframes[timeframe]
        if isinstance(form, str):
            return form.format(abs_delta)
        if delta == 0:
            key = 'zero'
        elif delta < 0:
            key = 'past'
        elif 'future-singular' not in form:
            key = 'future'
        elif 2 <= abs_delta % 10 <= 4 and (abs_delta % 100 < 10 or abs_delta % 100 >= 20):
            key = 'future-singular'
        else:
            key = 'future-paucal'
        form: str = form[key]
        return form.format(abs_delta)