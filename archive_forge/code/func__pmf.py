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
def _pmf(self, x, M, n, N, odds):
    x, M, n, N, odds = np.broadcast_arrays(x, M, n, N, odds)
    if x.size == 0:
        return np.empty_like(x)

    @np.vectorize
    def _pmf1(x, M, n, N, odds):
        urn = self.dist(N, n, M, odds, 1e-12)
        return urn.probability(x)
    return _pmf1(x, M, n, N, odds)