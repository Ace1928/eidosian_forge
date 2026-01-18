import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _make_interp_spline_full_matrix(x, y, k, t, bc_type):
    """ Construct the interpolating spline spl(x) = y with *full* linalg.

        Only useful for testing, do not call directly!
        This version is O(N**2) in memory and O(N**3) in flop count.
    """
    if bc_type is None or bc_type == 'not-a-knot':
        deriv_l, deriv_r = (None, None)
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = (bc_type, bc_type)
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError('Unknown boundary condition: %s' % bc_type) from e
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]
    n = x.size
    nt = t.size - k - 1
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]
    n = x.size
    nt = t.size - k - 1
    assert nt - n == nleft + nright
    intervals = cupy.empty_like(x, dtype=cupy.int64)
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, x, intervals, k, nt, False, x.shape[0]))
    dummy_c = cupy.empty((nt, 1), dtype=float)
    out = cupy.empty((len(x), prod(dummy_c.shape[1:])), dtype=dummy_c.dtype)
    num_c = prod(dummy_c.shape[1:])
    temp = cupy.empty(x.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
    d_boor_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, dummy_c, k, 0, x, intervals, out, temp, num_c, 0, x.shape[0]))
    A = cupy.zeros((nt, nt), dtype=float)
    offset = nleft
    for j in range(len(x)):
        left = intervals[j]
        A[j + offset, left - k:left + 1] = temp[j * (2 * k + 1):j * (2 * k + 1) + k + 1]
    intervals_bc = cupy.empty(1, dtype=cupy.int64)
    if nleft > 0:
        intervals_bc[0] = intervals[0]
        x0 = cupy.array([x[0]], dtype=x.dtype)
        for j, m in enumerate(deriv_l_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            A[j, left - k:left + 1] = temp[:k + 1]
    if nright > 0:
        intervals_bc[0] = intervals[-1]
        x0 = cupy.array([x[-1]], dtype=x.dtype)
        for j, m in enumerate(deriv_r_ords):
            d_boor_kernel((1,), (1,), (t, dummy_c, k, int(m), x0, intervals_bc, out, temp, num_c, 0, 1))
            left = intervals_bc[0]
            row = nleft + len(x) + j
            A[row, left - k:left + 1] = temp[:k + 1]
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)
    from cupy.linalg import solve
    coef = solve(A, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((nt,) + y.shape[1:]))
    return BSpline(t, coef, k)