import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
class Tweedie(Family):
    """
    Tweedie family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Tweedie family is the log link.
        Available links are log, Power and any aliases of power.
        See statsmodels.genmod.families.links for more information.
    var_power : float, optional
        The variance power. The default is 1.
    eql : bool
        If True, the Extended Quasi-Likelihood is used, else the
        likelihood is used.
        In both cases, for likelihood computations the var_power
        must be between 1 and 2.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Tweedie.link : a link instance
        The link function of the Tweedie instance
    Tweedie.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.Power
    Tweedie.var_power : float
        The power parameter of the variance function.

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    Loglikelihood function not implemented because of the complexity of
    calculating an infinite series of summations. The variance power can be
    estimated using the ``estimate_tweedie_power`` function that is part of the
    statsmodels.genmod.generalized_linear_model.GLM class.
    """
    links = [L.Log, L.Power]
    variance = V.Power(power=1.5)
    safe_links = [L.Log, L.Power]

    def __init__(self, link=None, var_power=1.0, eql=False, check_link=True):
        self.var_power = var_power
        self.eql = eql
        if eql and (var_power < 1 or var_power > 2):
            raise ValueError('Tweedie: if EQL=True then var_power must fall between 1 and 2')
        if link is None:
            link = L.Log()
        super().__init__(link=link, variance=V.Power(power=var_power * 1.0), check_link=check_link)

    def _resid_dev(self, endog, mu):
        """
        Tweedie deviance residuals

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
        When :math:`p = 1`,

        .. math::

            dev_i = \\mu_i

        when :math:`endog_i = 0` and

        .. math::

            dev_i = endog_i * \\log(endog_i / \\mu_i) + (\\mu_i - endog_i)

        otherwise.

        When :math:`p = 2`,

        .. math::

            dev_i =  (endog_i - \\mu_i) / \\mu_i - \\log(endog_i / \\mu_i)

        For all other p,

        .. math::

            dev_i = endog_i^{2 - p} / ((1 - p) * (2 - p)) -
                    endog_i * \\mu_i^{1 - p} / (1 - p) + \\mu_i^{2 - p} /
                    (2 - p)

        The deviance residual is then

        .. math::

            resid\\_dev_i = 2 * dev_i
        """
        p = self.var_power
        if p == 1:
            dev = np.where(endog == 0, mu, endog * np.log(endog / mu) + (mu - endog))
        elif p == 2:
            endog1 = self._clean(endog)
            dev = (endog - mu) / mu - np.log(endog1 / mu)
        else:
            dev = endog ** (2 - p) / ((1 - p) * (2 - p)) - endog * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)
        return 2 * dev

    def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
        """
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Tweedie distribution.

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
        If eql is True, the Extended Quasi-Likelihood is used.  At present,
        this method returns NaN if eql is False.  When the actual likelihood
        is implemented, it will be accessible by setting eql to False.

        References
        ----------
        R Kaas (2005).  Compound Poisson Distributions and GLM's -- Tweedie's
        Distribution.
        https://core.ac.uk/download/pdf/6347266.pdf#page=11

        JA Nelder, D Pregibon (1987).  An extended quasi-likelihood function.
        Biometrika 74:2, pp 221-232.  https://www.jstor.org/stable/2336136
        """
        p = self.var_power
        endog = np.atleast_1d(endog)
        if p == 1:
            return Poisson().loglike_obs(endog=endog, mu=mu, var_weights=var_weights, scale=scale)
        elif p == 2:
            return Gamma().loglike_obs(endog=endog, mu=mu, var_weights=var_weights, scale=scale)
        if not self.eql:
            if p < 1 or p > 2:
                return np.nan
            if SP_LT_17:
                return np.nan
            scale = scale / var_weights
            theta = mu ** (1 - p) / (1 - p)
            kappa = mu ** (2 - p) / (2 - p)
            alpha = (2 - p) / (1 - p)
            ll_obs = (endog * theta - kappa) / scale
            idx = endog > 0
            if np.any(idx):
                if not np.isscalar(endog):
                    endog = endog[idx]
                if not np.isscalar(scale):
                    scale = scale[idx]
                x = ((p - 1) * scale / endog) ** alpha
                x /= (2 - p) * scale
                wb = special.wright_bessel(-alpha, 0, x)
                ll_obs[idx] += np.log(1 / endog * wb)
            return ll_obs
        else:
            llf = np.log(2 * np.pi * scale) + p * np.log(endog)
            llf -= np.log(var_weights)
            llf /= -2
            u = endog ** (2 - p) - (2 - p) * endog * mu ** (1 - p) + (1 - p) * mu ** (2 - p)
            u *= var_weights / (scale * (1 - p) * (2 - p))
        return llf - u

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
            The Anscombe residuals as defined below.

        Notes
        -----
        When :math:`p = 3`, then

        .. math::

            resid\\_anscombe_i = \\log(endog_i / \\mu_i) / \\sqrt{\\mu_i * scale} *
            \\sqrt(var\\_weights)

        Otherwise,

        .. math::

            c = (3 - p) / 3

        .. math::

            resid\\_anscombe_i = (1 / c) * (endog_i^c - \\mu_i^c) / \\mu_i^{p / 6}
            / \\sqrt{scale} * \\sqrt(var\\_weights)
        """
        if self.var_power == 3:
            resid = np.log(endog / mu) / np.sqrt(mu * scale)
        else:
            c = (3.0 - self.var_power) / 3.0
            resid = 1.0 / c * (endog ** c - mu ** c) / mu ** (self.var_power / 6.0) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid