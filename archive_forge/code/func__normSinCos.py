import math
from typing import NamedTuple
from dataclasses import dataclass
def _normSinCos(v):
    if abs(v) < _EPSILON:
        v = 0
    elif v > _ONE_EPSILON:
        v = 1
    elif v < _MINUS_ONE_EPSILON:
        v = -1
    return v