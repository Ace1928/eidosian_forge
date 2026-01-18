import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def _do_scaling(self):
    arr = self._array
    out_dtype = self._out_dtype
    assert out_dtype.kind in 'iu'
    mn, mx = self.finite_range()
    if arr.dtype.kind == 'f':
        if self._nan2zero and self.has_nan:
            mn = min(mn, 0)
            mx = max(mx, 0)
        self._range_scale(mn, mx)
        return
    info = np.iinfo(out_dtype)
    out_max, out_min = (info.max, info.min)
    if int(mx) <= int(out_max) and int(mn) >= int(out_min):
        return
    self._iu2iu()