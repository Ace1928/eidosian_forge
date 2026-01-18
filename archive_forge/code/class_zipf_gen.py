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
class zipf_gen(rv_discrete):
    """A Zipf (Zeta) discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipfian

    Notes
    -----
    The probability mass function for `zipf` is:

    .. math::

        f(k, a) = \\frac{1}{\\zeta(a) k^a}

    for :math:`k \\ge 1`, :math:`a > 1`.

    `zipf` takes :math:`a > 1` as shape parameter. :math:`\\zeta` is the
    Riemann zeta function (`scipy.special.zeta`)

    The Zipf distribution is also known as the zeta distribution, which is
    a special case of the Zipfian distribution (`zipfian`).

    %(after_notes)s

    References
    ----------
    .. [1] "Zeta Distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Zeta_distribution

    %(example)s

    Confirm that `zipf` is the large `n` limit of `zipfian`.

    >>> import numpy as np
    >>> from scipy.stats import zipf, zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipf.pmf(k, a), zipfian.pmf(k, a, n=10000000))
    True

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (1, np.inf), (False, False))]

    def _rvs(self, a, size=None, random_state=None):
        return random_state.zipf(a, size=size)

    def _argcheck(self, a):
        return a > 1

    def _pmf(self, k, a):
        Pk = 1.0 / special.zeta(a, 1) / k ** a
        return Pk

    def _munp(self, n, a):
        return _lazywhere(a > n + 1, (a, n), lambda a, n: special.zeta(a - n, 1) / special.zeta(a, 1), np.inf)