import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _update_cupy_array(self):
    """
        Updates _cupy_array from _numpy_array.
        To be executed before calling cupy function.
        """
    base = self.base
    if base is None:
        if self._remember_numpy:
            if self._cupy_array is None:
                self._cupy_array = cp.array(self._numpy_array)
            else:
                self._cupy_array[:] = self._numpy_array
    elif base._remember_numpy:
        base._update_cupy_array()