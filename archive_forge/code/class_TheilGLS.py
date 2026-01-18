from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
class TheilGLS(GLS):
    """GLS with stochastic restrictions

    TheilGLS estimates the following linear model

    .. math:: y = X \\beta + u

    using additional information given by a stochastic constraint

    .. math:: q = R \\beta + v

    :math:`E(u) = 0`, :math:`cov(u) = \\Sigma`
    :math:`cov(u, v) = \\Sigma_p`, with full rank.

    u and v are assumed to be independent of each other.
    If :math:`E(v) = 0`, then the estimator is unbiased.

    Note: The explanatory variables are not rescaled, the parameter estimates
    not scale equivariant and fitted values are not scale invariant since
    scaling changes the relative penalization weights (for given \\Sigma_p).

    Note: GLS is not tested yet, only Sigma is identity is tested

    Notes
    -----

    The parameter estimates solves the moment equation:

    .. math:: (X' \\Sigma X + \\lambda R' \\sigma^2 \\Sigma_p^{-1} R) b = X' \\Sigma y + \\lambda R' \\Sigma_p^{-1} q

    :math:`\\lambda` is the penalization weight similar to Ridge regression.

    If lambda is zero, then the parameter estimate is the same as OLS. If
    lambda goes to infinity, then the restriction is imposed with equality.
    In the model `pen_weight` is used as name instead of $\\lambda$

    R does not have to be square. The number of rows of R can be smaller
    than the number of parameters. In this case not all linear combination
    of parameters are penalized.

    The stochastic constraint can be interpreted in several different ways:

     - The prior information represents parameter estimates from independent
       prior samples.
     - We can consider it just as linear restrictions that we do not want
       to impose without uncertainty.
     - With a full rank square restriction matrix R, the parameter estimate
       is the same as a Bayesian posterior mean for the case of an informative
       normal prior, normal likelihood and known error variance Sigma. If R
       is less than full rank, then it defines a partial prior.

    References
    ----------
    Theil Goldberger

    Baum, Christopher slides for tgmixed in Stata

    (I do not remember what I used when I first wrote the code.)

    Parameters
    ----------
    endog : array_like, 1-D
        dependent or endogenous variable
    exog : array_like, 1D or 2D
        array of explanatory or exogenous variables
    r_matrix : None or array_like, 2D
        array of linear restrictions for stochastic constraint.
        default is identity matrix that does not penalize constant, if constant
        is detected to be in `exog`.
    q_matrix : None or array_like
        mean of the linear restrictions. If None, the it is set to zeros.
    sigma_prior : None or array_like
        A fully specified sigma_prior is a square matrix with the same number
        of rows and columns as there are constraints (number of rows of r_matrix).
        If sigma_prior is None, a scalar or one-dimensional, then a diagonal matrix
        is created.
    sigma : None or array_like
        Sigma is the covariance matrix of the error term that is used in the same
        way as in GLS.
    """

    def __init__(self, endog, exog, r_matrix=None, q_matrix=None, sigma_prior=None, sigma=None):
        super().__init__(endog, exog, sigma=sigma)
        if r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
        else:
            try:
                const_idx = self.data.const_idx
            except AttributeError:
                const_idx = None
            k_exog = exog.shape[1]
            r_matrix = np.eye(k_exog)
            if const_idx is not None:
                keep_idx = lrange(k_exog)
                del keep_idx[const_idx]
                r_matrix = r_matrix[keep_idx]
        k_constraints, k_exog = r_matrix.shape
        self.r_matrix = r_matrix
        if k_exog != self.exog.shape[1]:
            raise ValueError('r_matrix needs to have the same number of columnsas exog')
        if q_matrix is not None:
            self.q_matrix = atleast_2dcols(q_matrix)
        else:
            self.q_matrix = np.zeros(k_constraints)[:, None]
        if self.q_matrix.shape != (k_constraints, 1):
            raise ValueError('q_matrix has wrong shape')
        if sigma_prior is not None:
            sigma_prior = np.asarray(sigma_prior)
            if np.size(sigma_prior) == 1:
                sigma_prior = np.diag(sigma_prior * np.ones(k_constraints))
            elif sigma_prior.ndim == 1:
                sigma_prior = np.diag(sigma_prior)
        else:
            sigma_prior = np.eye(k_constraints)
        if sigma_prior.shape != (k_constraints, k_constraints):
            raise ValueError('sigma_prior has wrong shape')
        self.sigma_prior = sigma_prior
        self.sigma_prior_inv = np.linalg.pinv(sigma_prior)

    def fit(self, pen_weight=1.0, cov_type='sandwich', use_t=True):
        """Estimate parameters and return results instance

        Parameters
        ----------
        pen_weight : float
            penalization factor for the restriction, default is 1.
        cov_type : str, 'data-prior' or 'sandwich'
            'data-prior' assumes that the stochastic restriction reflects a
            previous sample. The covariance matrix of the parameter estimate
            is in this case the same form as the one of GLS.
            The covariance matrix for cov_type='sandwich' treats the stochastic
            restriction (R and q) as fixed and has a sandwich form analogously
            to M-estimators.

        Returns
        -------
        results : TheilRegressionResults instance

        Notes
        -----
        cov_params for cov_type data-prior, is calculated as

        .. math:: \\sigma^2 A^{-1}

        cov_params for cov_type sandwich, is calculated as

        .. math:: \\sigma^2 A^{-1} (X'X) A^{-1}

        where :math:`A = X' \\Sigma X + \\lambda \\sigma^2 R' \\Simga_p^{-1} R`

        :math:`\\sigma^2` is an estimate of the error variance.
        :math:`\\sigma^2` inside A is replaced by the estimate from the initial
        GLS estimate. :math:`\\sigma^2` in cov_params is obtained from the
        residuals of the final estimate.

        The sandwich form of the covariance estimator is not robust to
        misspecified heteroscedasticity or autocorrelation.
        """
        lambd = pen_weight
        res_gls = GLS(self.endog, self.exog, sigma=self.sigma).fit()
        self.res_gls = res_gls
        sigma2_e = res_gls.mse_resid
        r_matrix = self.r_matrix
        q_matrix = self.q_matrix
        sigma_prior_inv = self.sigma_prior_inv
        x = self.wexog
        y = self.wendog[:, None]
        xx = np.dot(x.T, x)
        xpx = xx + sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, r_matrix))
        xpy = np.dot(x.T, y) + sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, q_matrix))
        xpxi = np.linalg.pinv(xpx, rcond=1e-15 ** 2)
        xpxi_sandwich = xpxi.dot(xx).dot(xpxi)
        params = np.dot(xpxi, xpy)
        params = np.squeeze(params)
        if cov_type == 'sandwich':
            normalized_cov_params = xpxi_sandwich
        elif cov_type == 'data-prior':
            normalized_cov_params = xpxi
        else:
            raise ValueError("cov_type has to be 'sandwich' or 'data-prior'")
        self.normalized_cov_params = xpxi_sandwich
        self.xpxi = xpxi
        self.sigma2_e = sigma2_e
        lfit = TheilRegressionResults(self, params, normalized_cov_params=normalized_cov_params, use_t=use_t)
        lfit.penalization_factor = lambd
        return lfit

    def select_pen_weight(self, method='aicc', start_params=1.0, optim_args=None):
        """find penalization factor that minimizes gcv or an information criterion

        Parameters
        ----------
        method : str
            the name of an attribute of the results class. Currently the following
            are available aic, aicc, bic, gc and gcv.
        start_params : float
            starting values for the minimization to find the penalization factor
            `lambd`. Not since there can be local minima, it is best to try
            different starting values.
        optim_args : None or dict
            optimization keyword arguments used with `scipy.optimize.fmin`

        Returns
        -------
        min_pen_weight : float
            The penalization factor at which the target criterion is (locally)
            minimized.

        Notes
        -----
        This uses `scipy.optimize.fmin` as optimizer.
        """
        if optim_args is None:
            optim_args = {}

        def get_ic(lambd):
            return getattr(self.fit(lambd), method)
        from scipy import optimize
        lambd = optimize.fmin(get_ic, start_params, **optim_args)
        return lambd