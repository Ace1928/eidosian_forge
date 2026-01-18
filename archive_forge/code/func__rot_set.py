from typing import (
from numbers import Integral, Real
def _rot_set(s: set, k: int, n: int):
    k %= n
    if not k:
        return s
    return {(v + k) % n for v in s}