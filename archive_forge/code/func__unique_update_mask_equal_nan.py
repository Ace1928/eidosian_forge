import numpy
import cupy
from cupy import _core
@_core.fusion.fuse()
def _unique_update_mask_equal_nan(mask, x0):
    mask1 = cupy.logical_not(cupy.isnan(x0))
    mask[:] = cupy.logical_and(mask, mask1)