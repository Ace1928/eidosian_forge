import numpy as np
from scipy import stats
from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.distributions.copula.copulas import Copula
class IndependenceCopula(Copula):
    """Independence copula.

    Copula with independent random variables.

    .. math::

        C_	heta(u,v) = uv

    Parameters
    ----------
    k_dim : int
        Dimension, number of components in the multivariate random variable.

    Notes
    -----
    IndependenceCopula does not have copula parameters.
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    """

    def __init__(self, k_dim=2):
        super().__init__(k_dim=k_dim)

    def _handle_args(self, args):
        if args != () and args is not None:
            msg = 'Independence copula does not use copula parameters.'
            raise ValueError(msg)
        else:
            return args

    def rvs(self, nobs=1, args=(), random_state=None):
        self._handle_args(args)
        rng = check_random_state(random_state)
        x = rng.random((nobs, self.k_dim))
        return x

    def pdf(self, u, args=()):
        u = np.asarray(u)
        return np.ones(u.shape[:-1])

    def cdf(self, u, args=()):
        return np.prod(u, axis=-1)

    def tau(self):
        return 0

    def plot_pdf(self, *args):
        raise NotImplementedError('PDF is constant over the domain.')