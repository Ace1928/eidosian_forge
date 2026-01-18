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
class nhypergeom_gen(rv_discrete):
    """A negative hypergeometric discrete random variable.

    Consider a box containing :math:`M` balls:, :math:`n` red and
    :math:`M-n` blue. We randomly sample balls from the box, one
    at a time and *without* replacement, until we have picked :math:`r`
    blue balls. `nhypergeom` is the distribution of the number of
    red balls :math:`k` we have picked.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `r`) are not
    universally accepted. See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: f(k; M, n, r) = \\frac{{{k+r-1}\\choose{k}}{{M-r-k}\\choose{n-k}}}
                                   {{M \\choose n}}

    for :math:`k \\in [0, n]`, :math:`n \\in [0, M]`, :math:`r \\in [0, M-n]`,
    and the binomial coefficient is:

    .. math:: \\binom{n}{k} \\equiv \\frac{n!}{k! (n - k)!}.

    It is equivalent to observing :math:`k` successes in :math:`k+r-1`
    samples with :math:`k+r`'th sample being a failure. The former
    can be modelled as a hypergeometric distribution. The probability
    of the latter is simply the number of failures remaining
    :math:`M-n-(r-1)` divided by the size of the remaining population
    :math:`M-(k+r-1)`. This relationship can be shown as:

    .. math:: NHG(k;M,n,r) = HG(k;M,n,k+r-1)\\frac{(M-n-(r-1))}{(M-(k+r-1))}

    where :math:`NHG` is probability mass function (PMF) of the
    negative hypergeometric distribution and :math:`HG` is the
    PMF of the hypergeometric distribution.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import nhypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.
    Then if we want to know the probability of finding a given number
    of dogs (successes) in a sample with exactly 12 animals that
    aren't dogs (failures), we can initialize a frozen distribution
    and plot the probability mass function:

    >>> M, n, r = [20, 7, 12]
    >>> rv = nhypergeom(M, n, r)
    >>> x = np.arange(0, n+2)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group with given 12 failures')
    >>> ax.set_ylabel('nhypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `nhypergeom`
    methods directly.  To for example obtain the probability mass
    function, use:

    >>> prb = nhypergeom.pmf(x, M, n, r)

    And to generate random numbers:

    >>> R = nhypergeom.rvs(M, n, r, size=10)

    To verify the relationship between `hypergeom` and `nhypergeom`, use:

    >>> from scipy.stats import hypergeom, nhypergeom
    >>> M, n, r = 45, 13, 8
    >>> k = 6
    >>> nhypergeom.pmf(k, M, n, r)
    0.06180776620271643
    >>> hypergeom.pmf(k, M, n, k+r-1) * (M - n - (r-1)) / (M - (k+r-1))
    0.06180776620271644

    See Also
    --------
    hypergeom, binom, nbinom

    References
    ----------
    .. [1] Negative Hypergeometric Distribution on Wikipedia
           https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution

    .. [2] Negative Hypergeometric Distribution from
           http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Negativehypergeometric.pdf

    """

    def _shape_info(self):
        return [_ShapeInfo('M', True, (0, np.inf), (True, False)), _ShapeInfo('n', True, (0, np.inf), (True, False)), _ShapeInfo('r', True, (0, np.inf), (True, False))]

    def _get_support(self, M, n, r):
        return (0, n)

    def _argcheck(self, M, n, r):
        cond = (n >= 0) & (n <= M) & (r >= 0) & (r <= M - n)
        cond &= _isintegral(M) & _isintegral(n) & _isintegral(r)
        return cond

    def _rvs(self, M, n, r, size=None, random_state=None):

        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, r, size, random_state):
            a, b = self.support(M, n, r)
            ks = np.arange(a, b + 1)
            cdf = self.cdf(ks, M, n, r)
            ppf = interp1d(cdf, ks, kind='next', fill_value='extrapolate')
            rvs = ppf(random_state.uniform(size=size)).astype(int)
            if size is None:
                return rvs.item()
            return rvs
        return _rvs1(M, n, r, size=size, random_state=random_state)

    def _logpmf(self, k, M, n, r):
        cond = (r == 0) & (k == 0)
        result = _lazywhere(~cond, (k, M, n, r), lambda k, M, n, r: -betaln(k + 1, r) + betaln(k + r, 1) - betaln(n - k + 1, M - r - n + 1) + betaln(M - r - k + 1, 1) + betaln(n + 1, M - n + 1) - betaln(M + 1, 1), fillvalue=0.0)
        return result

    def _pmf(self, k, M, n, r):
        return exp(self._logpmf(k, M, n, r))

    def _stats(self, M, n, r):
        M, n, r = (1.0 * M, 1.0 * n, 1.0 * r)
        mu = r * n / (M - n + 1)
        var = r * (M + 1) * n / ((M - n + 1) * (M - n + 2)) * (1 - r / (M - n + 1))
        g1, g2 = (None, None)
        return (mu, var, g1, g2)