import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def _writing_range(self):
    """Finite range for thresholding on write"""
    if self._out_dtype.kind in 'iu' and self._array.dtype.kind == 'f':
        mn, mx = self.finite_range()
        if (mn, mx) == (np.inf, -np.inf):
            mn, mx = (0, 0)
        return (mn, mx)
    return (None, None)