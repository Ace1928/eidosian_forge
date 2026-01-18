from typing import (
from numbers import Integral, Real
def _rot_list(l: list, k: int):
    """Rotate list by k items forward.  Ie. item at position 0 will be
    at position k in returned list.  Negative k is allowed."""
    n = len(l)
    k %= n
    if not k:
        return l
    return l[n - k:] + l[:n - k]