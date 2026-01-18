from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _distance_line_from_origin(line: Line) -> float:
    v = line['slope'] ** 2 + 1
    return abs(line['intercept']) / _math.sqrt(v)