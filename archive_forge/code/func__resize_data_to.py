import numbers
from functools import reduce
from operator import mul
import numpy as np
def _resize_data_to(self, n_rows, build_cache):
    """Resize data array if required"""
    n_bufs = np.ceil(n_rows / build_cache.rows_per_buf)
    extended_n_rows = int(n_bufs * build_cache.rows_per_buf)
    new_shape = (extended_n_rows,) + build_cache.common_shape
    if self._data.size == 0:
        self._data = np.empty(new_shape, dtype=build_cache.dtype)
    else:
        try:
            self._data.resize(new_shape)
        except ValueError:
            self._data = self._data.copy()
            self._data.resize(new_shape, refcheck=False)