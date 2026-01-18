import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def _iu2iu(self):
    mn, mx = (int(v) for v in self.finite_range())
    out_dtype = self._out_dtype
    o_min, o_max = (int(v) for v in shared_range(self.scaler_dtype, out_dtype))
    type_range = o_max - o_min
    mn2mx = mx - mn
    if mn2mx <= type_range:
        if o_min == 0:
            inter = floor_exact(mn - o_min, self.scaler_dtype)
        else:
            midpoint = mn + int(np.ceil(mn2mx / 2.0))
            inter = floor_exact(midpoint, self.scaler_dtype)
        int_inter = int(inter)
        assert mn - int_inter >= o_min
        if mx - int_inter <= o_max:
            self.inter = inter
            return
    super()._iu2iu()