from warnings import warn
import numpy as np
from numpy import (atleast_2d, arange, zeros_like, imag, diag,
from scipy._lib._util import ComplexWarning
from ._decomp import _asarray_validated
from .lapack import get_lapack_funcs, _compute_lwork
def _ldl_sanitize_ipiv(a, lower=True):
    """
    This helper function takes the rather strangely encoded permutation array
    returned by the LAPACK routines ?(HE/SY)TRF and converts it into
    regularized permutation and diagonal pivot size format.

    Since FORTRAN uses 1-indexing and LAPACK uses different start points for
    upper and lower formats there are certain offsets in the indices used
    below.

    Let's assume a result where the matrix is 6x6 and there are two 2x2
    and two 1x1 blocks reported by the routine. To ease the coding efforts,
    we still populate a 6-sized array and fill zeros as the following ::

        pivots = [2, 0, 2, 0, 1, 1]

    This denotes a diagonal matrix of the form ::

        [x x        ]
        [x x        ]
        [    x x    ]
        [    x x    ]
        [        x  ]
        [          x]

    In other words, we write 2 when the 2x2 block is first encountered and
    automatically write 0 to the next entry and skip the next spin of the
    loop. Thus, a separate counter or array appends to keep track of block
    sizes are avoided. If needed, zeros can be filtered out later without
    losing the block structure.

    Parameters
    ----------
    a : ndarray
        The permutation array ipiv returned by LAPACK
    lower : bool, optional
        The switch to select whether upper or lower triangle is chosen in
        the LAPACK call.

    Returns
    -------
    swap_ : ndarray
        The array that defines the row/column swap operations. For example,
        if row two is swapped with row four, the result is [0, 3, 2, 3].
    pivots : ndarray
        The array that defines the block diagonal structure as given above.

    """
    n = a.size
    swap_ = arange(n)
    pivots = zeros_like(swap_, dtype=int)
    skip_2x2 = False
    x, y, rs, re, ri = (1, 0, 0, n, 1) if lower else (-1, -1, n - 1, -1, -1)
    for ind in range(rs, re, ri):
        if skip_2x2:
            skip_2x2 = False
            continue
        cur_val = a[ind]
        if cur_val > 0:
            if cur_val != ind + 1:
                swap_[ind] = swap_[cur_val - 1]
            pivots[ind] = 1
        elif cur_val < 0 and cur_val == a[ind + x]:
            if -cur_val != ind + 2:
                swap_[ind + x] = swap_[-cur_val - 1]
            pivots[ind + y] = 2
            skip_2x2 = True
        else:
            raise ValueError('While parsing the permutation array in "scipy.linalg.ldl", invalid entries found. The array syntax is invalid.')
    return (swap_, pivots)