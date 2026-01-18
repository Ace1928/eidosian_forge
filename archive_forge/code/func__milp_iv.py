import warnings
import numpy as np
from scipy.sparse import csc_array, vstack, issparse
from scipy._lib._util import VisibleDeprecationWarning
from ._highs._highs_wrapper import _highs_wrapper  # type: ignore[import]
from ._constraints import LinearConstraint, Bounds
from ._optimize import OptimizeResult
from ._linprog_highs import _highs_to_scipy_status_message
def _milp_iv(c, integrality, bounds, constraints, options):
    if issparse(c):
        raise ValueError('`c` must be a dense array.')
    c = np.atleast_1d(c).astype(np.float64)
    if c.ndim != 1 or c.size == 0 or (not np.all(np.isfinite(c))):
        message = '`c` must be a one-dimensional array of finite numbers with at least one element.'
        raise ValueError(message)
    if issparse(integrality):
        raise ValueError('`integrality` must be a dense array.')
    message = '`integrality` must contain integers 0-3 and be broadcastable to `c.shape`.'
    if integrality is None:
        integrality = 0
    try:
        integrality = np.broadcast_to(integrality, c.shape).astype(np.uint8)
    except ValueError:
        raise ValueError(message)
    if integrality.min() < 0 or integrality.max() > 3:
        raise ValueError(message)
    if bounds is None:
        bounds = Bounds(0, np.inf)
    elif not isinstance(bounds, Bounds):
        message = '`bounds` must be convertible into an instance of `scipy.optimize.Bounds`.'
        try:
            bounds = Bounds(*bounds)
        except TypeError as exc:
            raise ValueError(message) from exc
    try:
        lb = np.broadcast_to(bounds.lb, c.shape).astype(np.float64)
        ub = np.broadcast_to(bounds.ub, c.shape).astype(np.float64)
    except (ValueError, TypeError) as exc:
        message = '`bounds.lb` and `bounds.ub` must contain reals and be broadcastable to `c.shape`.'
        raise ValueError(message) from exc
    if not constraints:
        constraints = [LinearConstraint(np.empty((0, c.size)), np.empty((0,)), np.empty((0,)))]
    try:
        A, b_l, b_u = _constraints_to_components(constraints)
    except ValueError as exc:
        message = '`constraints` (or each element within `constraints`) must be convertible into an instance of `scipy.optimize.LinearConstraint`.'
        raise ValueError(message) from exc
    if A.shape != (b_l.size, c.size):
        message = 'The shape of `A` must be (len(b_l), len(c)).'
        raise ValueError(message)
    indptr, indices, data = (A.indptr, A.indices, A.data.astype(np.float64))
    options = options or {}
    supported_options = {'disp', 'presolve', 'time_limit', 'node_limit', 'mip_rel_gap'}
    unsupported_options = set(options).difference(supported_options)
    if unsupported_options:
        message = f'Unrecognized options detected: {unsupported_options}. These will be passed to HiGHS verbatim.'
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    options_iv = {'log_to_console': options.pop('disp', False), 'mip_max_nodes': options.pop('node_limit', None)}
    options_iv.update(options)
    return (c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv)