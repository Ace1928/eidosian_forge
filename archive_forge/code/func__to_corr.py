import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _to_corr(self, m):
    """
        Given a psd matrix m, rotate to put one's on the diagonal, turning it
        into a correlation matrix.  This also requires the trace equal the
        dimensionality. Note: modifies input matrix
        """
    if not (m.flags.c_contiguous and m.dtype == np.float64 and (m.shape[0] == m.shape[1])):
        raise ValueError()
    d = m.shape[0]
    for i in range(d - 1):
        if m[i, i] == 1:
            continue
        elif m[i, i] > 1:
            for j in range(i + 1, d):
                if m[j, j] < 1:
                    break
        else:
            for j in range(i + 1, d):
                if m[j, j] > 1:
                    break
        c, s = self._givens_to_1(m[i, i], m[j, j], m[i, j])
        mv = m.ravel()
        drot(mv, mv, c, -s, n=d, offx=i * d, incx=1, offy=j * d, incy=1, overwrite_x=True, overwrite_y=True)
        drot(mv, mv, c, -s, n=d, offx=i, incx=d, offy=j, incy=d, overwrite_x=True, overwrite_y=True)
    return m