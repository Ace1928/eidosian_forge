from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
import numpy as np
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
import scipy.stats._boost as _boost
from ._biasedurn import (_PyFishersNCHypergeometric,
@_vectorize_rvs_over_shapes
def _rvs1(M, n, N, odds, size, random_state):
    length = np.prod(size)
    urn = _PyStochasticLib3()
    rv_gen = getattr(urn, self.rvs_name)
    rvs = rv_gen(N, n, M, odds, length, random_state)
    rvs = rvs.reshape(size)
    return rvs