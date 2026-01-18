import cupy
from cupy import _core
import cupy._core._routines_manipulation as _manipulation
def _atleast_nd_helper(n, arys):
    """Helper function for atleast_nd functions."""
    res = []
    for a in arys:
        a = cupy.asarray(a)
        if a.ndim < n:
            new_shape = _atleast_nd_shape_map[n, a.ndim](a.shape)
            a = a.reshape(*new_shape)
        res.append(a)
    if len(res) == 1:
        res, = res
    return res