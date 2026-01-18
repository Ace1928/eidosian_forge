from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def _length_of_ray_until_intersect(theta: float, line: Line) -> float:
    return line['intercept'] / (_math.sin(theta) - line['slope'] * _math.cos(theta))