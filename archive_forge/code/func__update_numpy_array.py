import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _update_numpy_array(self):
    """
        Updates _numpy_array from _cupy_array.
        To be executed before calling numpy function.
        """
    base = self.base
    _type = np.ndarray if self._supports_cupy else self._numpy_array.__class__
    if self._supports_cupy:
        if base is None:
            if not self._remember_numpy:
                if self._numpy_array is None:
                    self._numpy_array = cp.asnumpy(self._cupy_array)
                else:
                    self._cupy_array.get(out=self._numpy_array)
        elif not base._remember_numpy:
            base._update_numpy_array()
            if self._numpy_array is None:
                self._numpy_array = base._numpy_array.view(type=_type)
                self._numpy_array.shape = self._cupy_array.shape
                self._numpy_array.strides = self._cupy_array.strides
    elif base is not None:
        assert base._supports_cupy
        if not base._remember_numpy:
            base._update_numpy_array()