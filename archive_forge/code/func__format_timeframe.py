import sys
from math import trunc
from typing import (
def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
    """
        Amharic awares time frame format function, takes into account
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