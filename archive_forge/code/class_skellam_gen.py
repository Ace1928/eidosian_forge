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
class skellam_gen(rv_discrete):
    """A  Skellam discrete random variable.

    %(before_notes)s

    Notes
    -----
    Probability distribution of the difference of two correlated or
    uncorrelated Poisson random variables.

    Let :math:`k_1` and :math:`k_2` be two Poisson-distributed r.v. with
    expected values :math:`\\lambda_1` and :math:`\\lambda_2`. Then,
    :math:`k_1 - k_2` follows a Skellam distribution with parameters
    :math:`\\mu_1 = \\lambda_1 - \\rho \\sqrt{\\lambda_1 \\lambda_2}` and
    :math:`\\mu_2 = \\lambda_2 - \\rho \\sqrt{\\lambda_1 \\lambda_2}`, where
    :math:`\\rho` is the correlation coefficient between :math:`k_1` and
    :math:`k_2`. If the two Poisson-distributed r.v. are independent then
    :math:`\\rho = 0`.

    Parameters :math:`\\mu_1` and :math:`\\mu_2` must be strictly positive.

    For details see: https://en.wikipedia.org/wiki/Skellam_distribution

    `skellam` takes :math:`\\mu_1` and :math:`\\mu_2` as shape parameters.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('mu1', False, (0, np.inf), (False, False)), _ShapeInfo('mu2', False, (0, np.inf), (False, False))]

    def _rvs(self, mu1, mu2, size=None, random_state=None):
        n = size
        return random_state.poisson(mu1, n) - random_state.poisson(mu2, n)

    def _pmf(self, x, mu1, mu2):
        with np.errstate(over='ignore'):
            px = np.where(x < 0, _boost._ncx2_pdf(2 * mu2, 2 * (1 - x), 2 * mu1) * 2, _boost._ncx2_pdf(2 * mu1, 2 * (1 + x), 2 * mu2) * 2)
        return px

    def _cdf(self, x, mu1, mu2):
        x = floor(x)
        with np.errstate(over='ignore'):
            px = np.where(x < 0, _boost._ncx2_cdf(2 * mu2, -2 * x, 2 * mu1), 1 - _boost._ncx2_cdf(2 * mu1, 2 * (x + 1), 2 * mu2))
        return px

    def _stats(self, mu1, mu2):
        mean = mu1 - mu2
        var = mu1 + mu2
        g1 = mean / sqrt(var ** 3)
        g2 = 1 / var
        return (mean, var, g1, g2)