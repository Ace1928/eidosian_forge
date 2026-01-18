from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _from_linear(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * c ** (5 / 12) - 0.055