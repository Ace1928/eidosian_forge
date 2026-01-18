import numba
import numpy as np
@numba.jit(nopython=True)
def _rescale_and_lookup1d_function(data, scale, offset, lut, out):
    vmin, vmax = (0, lut.shape[0] - 1)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = (data[r, c] - offset) * scale
            val = min(max(val, vmin), vmax)
            out[r, c] = lut[int(val)]