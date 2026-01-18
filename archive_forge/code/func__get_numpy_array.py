import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _get_numpy_array(self):
    """
        Returns _numpy_array (ex: np.ndarray, numpy.ma.MaskedArray,
        numpy.chararray etc.) of ndarray object. And marks self(ndarray)
        and it's base (if exist) as numpy up-to-date.
        """
    base = self.base
    if base is not None and base._supports_cupy:
        base._remember_numpy = True
    if self._supports_cupy:
        self._remember_numpy = True
    return self._numpy_array