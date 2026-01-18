from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _l_to_y(l: float) -> float:
    if l <= 8:
        return _ref_y * l / _kappa
    return _ref_y * ((l + 16) / 116) ** 3