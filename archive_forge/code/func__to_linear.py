from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _to_linear(c: float) -> float:
    if c > 0.04045:
        return ((c + 0.055) / 1.055) ** 2.4
    return c / 12.92