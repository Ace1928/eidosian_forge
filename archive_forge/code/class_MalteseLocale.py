import sys
from math import trunc
from typing import (
class MalteseLocale(Locale):
    names = ['mt', 'mt-mt']
    past = '{0} ilu'
    future = 'fi {0}'
    and_word = 'u'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'issa', 'second': 'sekonda', 'seconds': '{0} sekondi', 'minute': 'minuta', 'minutes': '{0} minuti', 'hour': 'siegħa', 'hours': {'dual': '{0} sagħtejn', 'plural': '{0} sigħat'}, 'day': 'jum', 'days': {'dual': '{0} jumejn', 'plural': '{0} ijiem'}, 'week': 'ġimgħa', 'weeks': {'dual': '{0} ġimagħtejn', 'plural': '{0} ġimgħat'}, 'month': 'xahar', 'months': {'dual': '{0} xahrejn', 'plural': '{0} xhur'}, 'year': 'sena', 'years': {'dual': '{0} sentejn', 'plural': '{0} snin'}}
    month_names = ['', 'Jannar', 'Frar', 'Marzu', 'April', 'Mejju', 'Ġunju', 'Lulju', 'Awwissu', 'Settembru', 'Ottubru', 'Novembru', 'Diċembru']
    month_abbreviations = ['', 'Jan', 'Fr', 'Mar', 'Apr', 'Mejju', 'Ġun', 'Lul', 'Aw', 'Sett', 'Ott', 'Nov', 'Diċ']
    day_names = ['', 'It-Tnejn', 'It-Tlieta', 'L-Erbgħa', 'Il-Ħamis', 'Il-Ġimgħa', 'Is-Sibt', 'Il-Ħadd']
    day_abbreviations = ['', 'T', 'TL', 'E', 'Ħ', 'Ġ', 'S', 'Ħ']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        delta = abs(delta)
        if isinstance(form, Mapping):
            if delta == 2:
                form = form['dual']
            else:
                form = form['plural']
        return form.format(delta)