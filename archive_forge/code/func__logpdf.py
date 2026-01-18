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
def _logpdf(self, x, dim, mu, kappa):
    """Log of the von Mises-Fisher probability density function.

        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
    x = np.asarray(x)
    self._check_data_vs_dist(x, dim)
    dotproducts = np.einsum('i,...i->...', mu, x)
    return self._log_norm_factor(dim, kappa) + kappa * dotproducts