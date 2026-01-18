import warnings
import numpy as np
from scipy.sparse.linalg._interface import LinearOperator
from .utils import make_system
from scipy.linalg import get_lapack_funcs
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def _get_atol_rtol(name, b_norm, tol=_NoValue, atol=0.0, rtol=1e-05):
    """
    A helper function to handle tolerance deprecations and normalization
    """
    if tol is not _NoValue:
        msg = f"'scipy.sparse.linalg.{name}' keyword argument `tol` is deprecated in favor of `rtol` and will be removed in SciPy v1.14.0. Until then, if set, it will override `rtol`."
        warnings.warn(msg, category=DeprecationWarning, stacklevel=4)
        rtol = float(tol) if tol is not None else rtol
    if atol == 'legacy':
        msg = f"'scipy.sparse.linalg.{name}' called with `atol='legacy'`. This behavior is deprecated and will result in an error in SciPy v1.14.0. To preserve current behaviour, set `atol=0.0`."
        warnings.warn(msg, category=DeprecationWarning, stacklevel=4)
        atol = 0
    if atol is None:
        msg = f"'scipy.sparse.linalg.{name}' called without specifying `atol`. This behavior is deprecated and will result in an error in SciPy v1.14.0. To preserve current behaviour, set `atol=rtol`, or, to adopt the future default, set `atol=0.0`."
        warnings.warn(msg, category=DeprecationWarning, stacklevel=4)
        atol = rtol
    atol = max(float(atol), float(rtol) * float(b_norm))
    return (atol, rtol)