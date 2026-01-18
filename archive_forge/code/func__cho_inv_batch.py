import math
import numpy as np
from numpy import asarray_chkfinite, asarray
from numpy.lib import NumpyVersion
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state
from scipy.linalg.blas import drot
from scipy.linalg._misc import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _cho_inv_batch(a, check_finite=True):
    """
    Invert the matrices a_i, using a Cholesky factorization of A, where
    a_i resides in the last two dimensions of a and the other indices describe
    the index i.

    Overwrites the data in a.

    Parameters
    ----------
    a : array
        Array of matrices to invert, where the matrices themselves are stored
        in the last two dimensions.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        Array of inverses of the matrices ``a_i``.

    See Also
    --------
    scipy.linalg.cholesky : Cholesky factorization of a matrix

    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) < 2 or a1.shape[-2] != a1.shape[-1]:
        raise ValueError('expected square matrix in last two dimensions')
    potrf, potri = get_lapack_funcs(('potrf', 'potri'), (a1,))
    triu_rows, triu_cols = np.triu_indices(a.shape[-2], k=1)
    for index in np.ndindex(a1.shape[:-2]):
        a1[index], info = potrf(a1[index], lower=True, overwrite_a=False, clean=False)
        if info > 0:
            raise LinAlgError('%d-th leading minor not positive definite' % info)
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal potrf' % -info)
        a1[index], info = potri(a1[index], lower=True, overwrite_c=False)
        if info > 0:
            raise LinAlgError('the inverse could not be computed')
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal potrf' % -info)
        a1[index][triu_rows, triu_cols] = a1[index][triu_cols, triu_rows]
    return a1