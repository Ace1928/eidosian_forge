import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula
class GaussianCopula(EllipticalCopula):
    """Gaussian copula.

    It is constructed from a multivariate normal distribution over
    :math:`\\mathbb{R}^d` by using the probability integral transform.

    For a given correlation matrix :math:`R \\in[-1, 1]^{d \\times d}`,
    the Gaussian copula with parameter matrix :math:`R` can be written
    as:

    .. math::

        C_R^{\\text{Gauss}}(u) = \\Phi_R\\left(\\Phi^{-1}(u_1),\\dots,
        \\Phi^{-1}(u_d) \\right),

    where :math:`\\Phi^{-1}` is the inverse cumulative distribution function
    of a standard normal and :math:`\\Phi_R` is the joint cumulative
    distribution function of a multivariate normal distribution with mean
    vector zero and covariance matrix equal to the correlation
    matrix :math:`R`.

    Parameters
    ----------
    corr : scalar or array_like
        Correlation or scatter matrix for the elliptical copula. In the
        bivariate case, ``corr` can be a scalar and is then considered as
        the correlation coefficient. If ``corr`` is None, then the scatter
        matrix is the identity matrix.
    k_dim : int
        Dimension, number of components in the multivariate random variable.
    allow_singular : bool
        Allow singular correlation matrix.
        The behavior when the correlation matrix is singular is determined by
        `scipy.stats.multivariate_normal`` and might not be appropriate for
        all copula or copula distribution metnods. Behavior might change in
        future versions.

    Notes
    -----
    Elliptical copulas require that copula parameters are set when the
    instance is created. Those parameters currently cannot be provided in the
    call to methods. (This will most likely change in future versions.)
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    References
    ----------
    .. [1] Joe, Harry, 2014, Dependence modeling with copulas. CRC press.
        p. 163

    """

    def __init__(self, corr=None, k_dim=2, allow_singular=False):
        super().__init__(k_dim=k_dim)
        if corr is None:
            corr = np.eye(k_dim)
        elif k_dim == 2 and np.size(corr) == 1:
            corr = np.array([[1.0, corr], [corr, 1.0]])
        self.corr = np.asarray(corr)
        self.args = (self.corr,)
        self.distr_uv = stats.norm
        self.distr_mv = stats.multivariate_normal(cov=corr, allow_singular=allow_singular)

    def dependence_tail(self, corr=None):
        """
        Bivariate tail dependence parameter.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : any
            Tail dependence for Gaussian copulas is always zero.
            Argument will be ignored

        Returns
        -------
        Lower and upper tail dependence coefficients of the copula with given
        Pearson correlation coefficient.
        """
        return (0, 0)

    def _arg_from_tau(self, tau):
        return self.corr_from_tau(tau)