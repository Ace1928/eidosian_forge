import warnings
import cupy
from cupy import _core
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content
def _check_interpolation_as_method(method, interpolation, fname):
    warnings.warn(f"the `interpolation=` argument to {fname} was renamed to `method=`, which has additional options.\nUsers of the modes 'nearest', 'lower', 'higher', or 'midpoint' are encouraged to review the method they. (Deprecated NumPy 1.22)", DeprecationWarning, stacklevel=3)
    if method != 'linear':
        raise TypeError('You shall not pass both `method` and `interpolation`!\n(`interpolation` is Deprecated in favor of `method`)')
    return interpolation