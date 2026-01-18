import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
class InverseGaussian(Family):
    """
    InverseGaussian exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the inverse Gaussian family is the
        inverse squared link.
        Available links are InverseSquared, Inverse, Log, and Identity.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    InverseGaussian.link : a link instance
        The link function of the inverse Gaussian instance
    InverseGaussian.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.mu_cubed

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    The inverse Gaussian distribution is sometimes referred to in the
    literature as the Wald distribution.
    """
    links = [L.InverseSquared, L.InversePower, L.Identity, L.Log]
    variance = V.mu_cubed
    safe_links = [L.InverseSquared, L.Log]

    def __init__(self, link=None, check_link=True):
        if link is None:
            link = L.InverseSquared()
        super().__init__(link=link, variance=InverseGaussian.variance, check_link=check_link)

    def _resid_dev(self, endog, mu):
        """
        Inverse Gaussian deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

           resid\\_dev_i = 1 / (endog_i * \\mu_i^2) * (endog_i - \\mu_i)^2
        """
        return 1.0 / (endog * mu ** 2) * (endog - mu) ** 2

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        """
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Inverse Gaussian distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        .. math::

           ll_i = -1/2 * (var\\_weights_i * (endog_i - \\mu_i)^2 /
           (scale * endog_i * \\mu_i^2) + \\ln(scale * \\endog_i^3 /
           var\\_weights_i) - \\ln(2 * \\pi))
        """
        ll_obs = -var_weights * (endog - mu) ** 2 / (scale * endog * mu ** 2)
        ll_obs += -np.log(scale * endog ** 3 / var_weights) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1.0, scale=1.0):
        """
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals for the inverse Gaussian distribution  as
            defined below

        Notes
        -----
        .. math::

           resid\\_anscombe_i = \\log(Y_i / \\mu_i) / \\sqrt{\\mu_i * scale} *
           \\sqrt(var\\_weights)
        """
        resid = np.log(endog / mu) / np.sqrt(mu * scale)
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.0):
        """
        Frozen Inverse Gaussian distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is required argument for get_distribution.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.

        Returns
        -------
        distribution instance

        """
        scale_ = scale / var_weights
        mu_ig = mu * scale_
        return stats.invgauss(mu_ig, scale=1 / scale_)