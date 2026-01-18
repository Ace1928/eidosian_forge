from __future__ import annotations
from typing import Literal
import numpy as np
from pandas.compat import pa_version_under10p1
def _str_get(self, i: int):
    lengths = pc.utf8_length(self._pa_array)
    if i >= 0:
        out_of_bounds = pc.greater_equal(i, lengths)
        start = i
        stop = i + 1
        step = 1
    else:
        out_of_bounds = pc.greater(-i, lengths)
        start = i
        stop = i - 1
        step = -1
    not_out_of_bounds = pc.invert(out_of_bounds.fill_null(True))
    selected = pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step)
    null_value = pa.scalar(None, type=self._pa_array.type)
    result = pc.if_else(not_out_of_bounds, selected, null_value)
    return type(self)(result)