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
def _lnB(alpha):
    """Internal helper function to compute the log of the useful quotient.

    .. math::

        B(\\alpha) = \\frac{\\prod_{i=1}{K}\\Gamma(\\alpha_i)}
                         {\\Gamma\\left(\\sum_{i=1}^{K} \\alpha_i \\right)}

    Parameters
    ----------
    %(_dirichlet_doc_default_callparams)s

    Returns
    -------
    B : scalar
        Helper quotient, internal use only

    """
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))