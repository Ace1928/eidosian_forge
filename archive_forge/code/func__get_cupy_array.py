import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _get_cupy_array(self):
    """
        Returns _cupy_array (cupy.ndarray) of ndarray object. And marks
        self(ndarray) and it's base (if exist) as numpy not up-to-date.
        """
    base = self.base
    if base is not None:
        base._remember_numpy = False
    self._remember_numpy = False
    return self._cupy_array