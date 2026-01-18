import sys
from math import trunc
from typing import (
class SlavicBaseLocale(Locale):
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]]

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        delta = abs(delta)
        if isinstance(form, Mapping):
            if delta % 10 == 1 and delta % 100 != 11:
                form = form['singular']
            elif 2 <= delta % 10 <= 4 and (delta % 100 < 10 or delta % 100 >= 20):
                form = form['dual']
            else:
                form = form['plural']
        return form.format(delta)