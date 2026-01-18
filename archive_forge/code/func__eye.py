import numbers
import math
import numpy as np
from scipy._lib._util import check_random_state, rng_integers
from ._sputils import upcast, get_index_dtype, isscalarlike
from ._sparsetools import csr_hstack
from ._bsr import bsr_matrix, bsr_array
from ._coo import coo_matrix, coo_array
from ._csc import csc_matrix, csc_array
from ._csr import csr_matrix, csr_array
from ._dia import dia_matrix, dia_array
from ._base import issparse, sparray
def _eye(m, n, k, dtype, format, as_sparray=True):
    if as_sparray:
        csr_sparse = csr_array
        csc_sparse = csc_array
        coo_sparse = coo_array
        diags_sparse = diags_array
    else:
        csr_sparse = csr_matrix
        csc_sparse = csc_matrix
        coo_sparse = coo_matrix
        diags_sparse = diags
    if n is None:
        n = m
    m, n = (int(m), int(n))
    if m == n and k == 0:
        if format in ['csr', 'csc']:
            idx_dtype = get_index_dtype(maxval=n)
            indptr = np.arange(n + 1, dtype=idx_dtype)
            indices = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            cls = {'csr': csr_sparse, 'csc': csc_sparse}[format]
            return cls((data, indices, indptr), (n, n))
        elif format == 'coo':
            idx_dtype = get_index_dtype(maxval=n)
            row = np.arange(n, dtype=idx_dtype)
            col = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            return coo_sparse((data, (row, col)), (n, n))
    data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)