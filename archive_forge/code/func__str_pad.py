from __future__ import annotations
from typing import Literal
import numpy as np
from pandas.compat import pa_version_under10p1
def _str_pad(self, width: int, side: Literal['left', 'right', 'both']='left', fillchar: str=' '):
    if side == 'left':
        pa_pad = pc.utf8_lpad
    elif side == 'right':
        pa_pad = pc.utf8_rpad
    elif side == 'both':
        pa_pad = pc.utf8_center
    else:
        raise ValueError(f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'")
    return type(self)(pa_pad(self._pa_array, width=width, padding=fillchar))