import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
def _make_design_matrix(x, t, k, extrapolate, indices):
    """
    Returns a design matrix in CSR format.
    Note that only indices is passed, but not indptr because indptr is already
    precomputed in the calling Python function design_matrix.

    Parameters
    ----------
    x : array_like, shape (n,)
        Points to evaluate the spline at.
    t : array_like, shape (nt,)
        Sorted 1D array of knots.
    k : int
        B-spline degree.
    extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points.
    indices : ndarray, shape (n * (k + 1),)
        Preallocated indices of the final CSR array.
    Returns
    -------
    data
        The data array of a CSR array of the b-spline design matrix.
        In each row all the basis elements are evaluated at the certain point
        (first row - x[0], ..., last row - x[-1]).

    indices
        The indices array of a CSR array of the b-spline design matrix.
    """
    n = t.shape[0] - k - 1
    intervals = cupy.empty_like(x, dtype=cupy.int64)
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, x, intervals, k, n, extrapolate, x.shape[0]))
    bspline_basis = cupy.empty(x.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', x)
    d_boor_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (t, None, k, 0, x, intervals, None, bspline_basis, 0, 0, x.shape[0]))
    data = cupy.zeros(x.shape[0] * (k + 1), dtype=cupy.float_)
    design_mat_kernel = _get_module_func(DESIGN_MAT_MODULE, 'compute_design_matrix', indices)
    design_mat_kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (k, intervals, bspline_basis, data, indices, x.shape[0]))
    return (data, indices)