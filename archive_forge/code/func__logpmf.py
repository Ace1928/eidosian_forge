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
def _logpmf(self, x, M, m, n, mxcond, ncond):
    num = np.zeros_like(m, dtype=np.float64)
    den = np.zeros_like(n, dtype=np.float64)
    m, x = (m[~mxcond], x[~mxcond])
    M, n = (M[~ncond], n[~ncond])
    num[~mxcond] = betaln(m + 1, 1) - betaln(x + 1, m - x + 1)
    den[~ncond] = betaln(M + 1, 1) - betaln(n + 1, M - n + 1)
    num[mxcond] = np.nan
    den[ncond] = np.nan
    num = num.sum(axis=-1)
    return num - den