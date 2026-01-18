import sys
from math import trunc
from typing import (
class IcelandicLocale(Locale):

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        if isinstance(form, Mapping):
            if delta < 0:
                form = form['past']
            elif delta > 0:
                form = form['future']
            else:
                raise ValueError('Icelandic Locale does not support units with a delta of zero. Please consider making a contribution to fix this issue.')
        return form.format(abs(delta))
    names = ['is', 'is-is']
    past = 'fyrir {0} síðan'
    future = 'eftir {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'rétt í þessu', 'second': {'past': 'sekúndu', 'future': 'sekúndu'}, 'seconds': {'past': '{0} nokkrum sekúndum', 'future': 'nokkrar sekúndur'}, 'minute': {'past': 'einni mínútu', 'future': 'eina mínútu'}, 'minutes': {'past': '{0} mínútum', 'future': '{0} mínútur'}, 'hour': {'past': 'einum tíma', 'future': 'einn tíma'}, 'hours': {'past': '{0} tímum', 'future': '{0} tíma'}, 'day': {'past': 'einum degi', 'future': 'einn dag'}, 'days': {'past': '{0} dögum', 'future': '{0} daga'}, 'month': {'past': 'einum mánuði', 'future': 'einn mánuð'}, 'months': {'past': '{0} mánuðum', 'future': '{0} mánuði'}, 'year': {'past': 'einu ári', 'future': 'eitt ár'}, 'years': {'past': '{0} árum', 'future': '{0} ár'}}
    meridians = {'am': 'f.h.', 'pm': 'e.h.', 'AM': 'f.h.', 'PM': 'e.h.'}
    month_names = ['', 'janúar', 'febrúar', 'mars', 'apríl', 'maí', 'júní', 'júlí', 'ágúst', 'september', 'október', 'nóvember', 'desember']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'maí', 'jún', 'júl', 'ágú', 'sep', 'okt', 'nóv', 'des']
    day_names = ['', 'mánudagur', 'þriðjudagur', 'miðvikudagur', 'fimmtudagur', 'föstudagur', 'laugardagur', 'sunnudagur']
    day_abbreviations = ['', 'mán', 'þri', 'mið', 'fim', 'fös', 'lau', 'sun']