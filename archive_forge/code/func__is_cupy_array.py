from __future__ import annotations
import sys
import math
def _is_cupy_array(x):
    if 'cupy' not in sys.modules:
        return False
    import cupy as cp
    return isinstance(x, (cp.ndarray, cp.generic))