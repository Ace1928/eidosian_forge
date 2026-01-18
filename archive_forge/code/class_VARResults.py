from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
class VARResults(VARProcess):
    """Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    sigma_u : ndarray
    lag_order : int
    model : VAR model instance
    trend : str {'n', 'c', 'ct'}
    names : array_like
        List of names of the endogenous variables in order of appearance in
        `endog`.
    dates
    exog : ndarray

    Attributes
    ----------
    params : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    dates
    endog
    endog_lagged
    k_ar : int
        Order of VAR process
    k_trend : int
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    names : list
        variables names
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    """
    _model_type = 'VAR'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order, model=None, trend='c', names=None, dates=None, exog=None):
        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates
        self.n_totobs, neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        self.trend = trend
        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(names, lag_order, k_trend, model.data.orig_exog)
        self.params = params
        self.exog = exog
        endog_start = k_trend
        if exog is not None:
            k_exog_user = exog.shape[1]
            endog_start += k_exog_user
        else:
            k_exog_user = 0
        reshaped = self.params[endog_start:]
        reshaped = reshaped.reshape((lag_order, neqs, neqs))
        coefs = reshaped.swapaxes(1, 2).copy()
        self.coefs_exog = params[:endog_start].T
        self.k_exog = self.coefs_exog.shape[1]
        self.k_exog_user = k_exog_user
        _params_info = {'k_trend': k_trend, 'k_exog_user': k_exog_user, 'k_ar': lag_order}
        super().__init__(coefs, self.coefs_exog, sigma_u, names=names, _params_info=_params_info)

    def plot(self):
        """Plot input time series"""
        return plotting.plot_mts(self.endog, names=self.names, index=self.dates)

    @property
    def df_model(self):
        """
        Number of estimated parameters per variable, including the intercept / trends
        """
        return self.neqs * self.k_ar + self.k_exog

    @property
    def df_resid(self):
        """Number of observations minus number of estimated parameters"""
        return self.nobs - self.df_model

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted insample values of the response variables of the model.
        """
        return np.dot(self.endog_lagged, self.params)

    @cache_readonly
    def resid(self):
        """
        Residuals of response variable resulting from estimated coefficients
        """
        return self.endog[self.k_ar:] - self.fittedvalues

    def sample_acov(self, nlags=1):
        """Sample acov"""
        return _compute_acov(self.endog[self.k_ar:], nlags=nlags)

    def sample_acorr(self, nlags=1):
        """Sample acorr"""
        acovs = self.sample_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    def plot_sample_acorr(self, nlags=10, linewidth=8):
        """
        Plot sample autocorrelation function

        Parameters
        ----------
        nlags : int
            The number of lags to use in compute the autocorrelation. Does
            not count the zero lag, which will be returned.
        linewidth : int
            The linewidth for the plots.

        Returns
        -------
        Figure
            The figure that contains the plot axes.
        """
        fig = plotting.plot_full_acorr(self.sample_acorr(nlags=nlags), linewidth=linewidth)
        return fig

    def resid_acov(self, nlags=1):
        """
        Compute centered sample autocovariance (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        """
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags=1):
        """
        Compute sample autocorrelation (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        """
        acovs = self.resid_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    @cache_readonly
    def resid_corr(self):
        """
        Centered residual correlation matrix
        """
        return self.resid_acorr(0)[0]

    @cache_readonly
    def sigma_u_mle(self):
        """(Biased) maximum likelihood estimate of noise process covariance"""
        if not self.df_resid:
            return np.zeros_like(self.sigma_u)
        return self.sigma_u * self.df_resid / self.nobs

    def cov_params(self):
        """Estimated variance-covariance of model coefficients

        Notes
        -----
        Covariance of vec(B), where B is the matrix
        [params_for_deterministic_terms, A_1, ..., A_p] with the shape
        (K x (Kp + number_of_deterministic_terms))
        Adjusted to be an unbiased estimator
        Ref: Lütkepohl p.74-75
        """
        z = self.endog_lagged
        return np.kron(np.linalg.inv(z.T @ z), self.sigma_u)

    def cov_ybar(self):
        """Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \\sqrt(T) (\\bar{y} - \\mu) \\rightarrow
                  {\\cal N}(0, \\Sigma_{\\bar{y}}) \\\\

            \\Sigma_{\\bar{y}} = B \\Sigma_u B^\\prime, \\text{where }
                  B = (I_K - A_1 - \\cdots - A_p)^{-1}

        Notes
        -----
        Lütkepohl Proposition 3.3
        """
        Ainv = np.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return Ainv @ self.sigma_u @ Ainv.T

    @cache_readonly
    def _zz(self):
        return np.dot(self.endog_lagged.T, self.endog_lagged)

    @property
    def _cov_alpha(self):
        """
        Estimated covariance matrix of model coefficients w/o exog
        """
        kn = self.k_exog * self.neqs
        return self.cov_params()[kn:, kn:]

    @cache_readonly
    def _cov_sigma(self):
        """
        Estimated covariance matrix of vech(sigma_u)
        """
        D_K = tsa.duplication_matrix(self.neqs)
        D_Kinv = np.linalg.pinv(D_K)
        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * D_Kinv @ sigxsig @ D_Kinv.T

    @cache_readonly
    def llf(self):
        """Compute VAR(p) loglikelihood"""
        return var_loglike(self.resid, self.sigma_u_mle, self.nobs)

    @cache_readonly
    def stderr(self):
        """Standard errors of coefficients, reshaped to match in size"""
        stderr = np.sqrt(np.diag(self.cov_params()))
        return stderr.reshape((self.df_model, self.neqs), order='C')
    bse = stderr

    @cache_readonly
    def stderr_endog_lagged(self):
        """Stderr_endog_lagged"""
        start = self.k_exog
        return self.stderr[start:]

    @cache_readonly
    def stderr_dt(self):
        """Stderr_dt"""
        end = self.k_exog
        return self.stderr[:end]

    @cache_readonly
    def tvalues(self):
        """
        Compute t-statistics. Use Student-t(T - Kp - 1) = t(df_resid) to
        test significance.
        """
        return self.params / self.stderr

    @cache_readonly
    def tvalues_endog_lagged(self):
        """tvalues_endog_lagged"""
        start = self.k_exog
        return self.tvalues[start:]

    @cache_readonly
    def tvalues_dt(self):
        """tvalues_dt"""
        end = self.k_exog
        return self.tvalues[:end]

    @cache_readonly
    def pvalues(self):
        """
        Two-sided p-values for model coefficients from Student t-distribution
        """
        return 2 * stats.norm.sf(np.abs(self.tvalues))

    @cache_readonly
    def pvalues_endog_lagged(self):
        """pvalues_endog_laggd"""
        start = self.k_exog
        return self.pvalues[start:]

    @cache_readonly
    def pvalues_dt(self):
        """pvalues_dt"""
        end = self.k_exog
        return self.pvalues[:end]

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True):
        """
        Plot forecast
        """
        mid, lower, upper = self.forecast_interval(self.endog[-self.k_ar:], steps, alpha=alpha)
        fig = plotting.plot_var_forc(self.endog, mid, lower, upper, names=self.names, plot_stderr=plot_stderr)
        return fig

    def forecast_cov(self, steps=1, method='mse'):
        """Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------
        steps : int

        Notes
        -----
        .. math:: \\Sigma_{\\hat y}(h) = \\Sigma_y(h) + \\Omega(h) / T

        Ref: Lütkepohl pp. 96-97

        Returns
        -------
        covs : ndarray (steps x k x k)
        """
        fc_cov = self.mse(steps)
        if method == 'mse':
            pass
        elif method == 'auto':
            if self.k_exog == 1 and self.k_trend < 2:
                fc_cov += self._omega_forc_cov(steps) / self.nobs
                import warnings
                warnings.warn('forecast cov takes parameter uncertainty intoaccount', OutputWarning, stacklevel=2)
        else:
            raise ValueError("method has to be either 'mse' or 'auto'")
        return fc_cov

    def irf_errband_mc(self, orth=False, repl=1000, steps=10, signif=0.05, seed=None, burn=100, cum=False):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        ma_coll = self.irf_resim(orth=orth, repl=repl, steps=steps, seed=seed, burn=burn, cum=cum)
        ma_sort = np.sort(ma_coll, axis=0)
        low_idx = int(round(signif / 2 * repl) - 1)
        upp_idx = int(round((1 - signif / 2) * repl) - 1)
        lower = ma_sort[low_idx, :, :, :]
        upper = ma_sort[upp_idx, :, :, :]
        return (lower, upper)

    def irf_resim(self, orth=False, repl=1000, steps=10, seed=None, burn=100, cum=False):
        """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        .. [*] Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse
           Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions
        """
        neqs = self.neqs
        k_ar = self.k_ar
        coefs = self.coefs
        sigma_u = self.sigma_u
        intercept = self.intercept
        nobs = self.nobs
        nobs_original = nobs + k_ar
        ma_coll = np.zeros((repl, steps + 1, neqs, neqs))

        def fill_coll(sim):
            ret = VAR(sim, exog=self.exog).fit(maxlags=k_ar, trend=self.trend)
            ret = ret.orth_ma_rep(maxn=steps) if orth else ret.ma_rep(maxn=steps)
            return ret.cumsum(axis=0) if cum else ret
        for i in range(repl):
            sim = util.varsim(coefs, intercept, sigma_u, seed=seed, steps=nobs_original + burn)
            sim = sim[burn:]
            ma_coll[i, :, :, :] = fill_coll(sim)
        return ma_coll

    def _omega_forc_cov(self, steps):
        G = self._zz
        Ginv = np.linalg.inv(G)
        B = self._bmat_forc_cov()
        _B = {}

        def bpow(i):
            if i not in _B:
                _B[i] = np.linalg.matrix_power(B, i)
            return _B[i]
        phis = self.ma_rep(steps)
        sig_u = self.sigma_u
        omegas = np.zeros((steps, self.neqs, self.neqs))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h - 1] = self.df_model * self.sigma_u
                continue
            om = omegas[h - 1]
            for i in range(h):
                for j in range(h):
                    Bi = bpow(h - 1 - i)
                    Bj = bpow(h - 1 - j)
                    mult = np.trace(Bi.T @ Ginv @ Bj @ G)
                    om += mult * phis[i] @ sig_u @ phis[j].T
            omegas[h - 1] = om
        return omegas

    def _bmat_forc_cov(self):
        upper = np.zeros((self.k_exog, self.df_model))
        upper[:, :self.k_exog] = np.eye(self.k_exog)
        lower_dim = self.neqs * (self.k_ar - 1)
        eye = np.eye(lower_dim)
        lower = np.column_stack((np.zeros((lower_dim, self.k_exog)), eye, np.zeros((lower_dim, self.neqs))))
        return np.vstack((upper, self.params.T, lower))

    def summary(self):
        """Compute console output summary of estimates

        Returns
        -------
        summary : VARSummary
        """
        return VARSummary(self)

    def irf(self, periods=10, var_decomp=None, var_order=None):
        """Analyze impulse responses to shocks in system

        Parameters
        ----------
        periods : int
        var_decomp : ndarray (k x k), lower triangular
            Must satisfy Omega = P P', where P is the passed matrix. Defaults
            to Cholesky decomposition of Omega
        var_order : sequence
            Alternate variable order for Cholesky decomposition

        Returns
        -------
        irf : IRAnalysis
        """
        if var_order is not None:
            raise NotImplementedError('alternate variable order not implemented (yet)')
        return IRAnalysis(self, P=var_decomp, periods=periods)

    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition ("fevd")

        Returns
        -------
        fevd : FEVD instance
        """
        return FEVD(self, P=var_decomp, periods=periods)

    def reorder(self, order):
        """Reorder variables for structural specification"""
        if len(order) != len(self.params[0, :]):
            raise ValueError('Reorder specification length should match number of endogenous variables')
        if isinstance(order[0], str):
            order_new = []
            for i, nam in enumerate(order):
                order_new.append(self.names.index(order[i]))
            order = order_new
        return _reordered(self, order)

    def test_causality(self, caused, causing=None, kind='f', signif=0.05):
        """
        Test Granger causality

        Parameters
        ----------
        caused : int or str or sequence of int or str
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-caused by the variable(s) specified
            by `causing`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-caused by the variable(s) specified
            by `causing`.
        causing : int or str or sequence of int or str or None, default: None
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-causing the variable(s) specified by
            `caused`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-causing the variable(s) specified by
            `caused`.
            If None, `causing` is assumed to be the complement of `caused`.
        kind : {'f', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables. The degrees of freedom in the F-test are based on the
        number of variables in the VAR system, that is, degrees of freedom
        are equal to the number of equations in the VAR times degree of freedom
        of a single equation.

        Test for Granger-causality as described in chapter 7.6.3 of [1]_.
        Test H0: "`causing` does not Granger-cause the remaining variables of
        the system" against  H1: "`causing` is Granger-causal for the
        remaining variables".

        Returns
        -------
        results : CausalityTestResults

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        if not 0 < signif < 1:
            raise ValueError('signif has to be between 0 and 1')
        allowed_types = (str, int)
        if isinstance(caused, allowed_types):
            caused = [caused]
        if not all((isinstance(c, allowed_types) for c in caused)):
            raise TypeError('caused has to be of type string or int (or a sequence of these types).')
        caused = [self.names[c] if type(c) is int else c for c in caused]
        caused_ind = [util.get_index(self.names, c) for c in caused]
        if causing is not None:
            if isinstance(causing, allowed_types):
                causing = [causing]
            if not all((isinstance(c, allowed_types) for c in causing)):
                raise TypeError('causing has to be of type string or int (or a sequence of these types) or None.')
            causing = [self.names[c] if type(c) is int else c for c in causing]
            causing_ind = [util.get_index(self.names, c) for c in causing]
        else:
            causing_ind = [i for i in range(self.neqs) if i not in caused_ind]
            causing = [self.names[c] for c in caused_ind]
        k, p = (self.neqs, self.k_ar)
        if p == 0:
            err = 'Cannot test Granger Causality in a model with 0 lags.'
            raise RuntimeError(err)
        num_restr = len(causing) * len(caused) * p
        num_det_terms = self.k_exog
        C = np.zeros((num_restr, k * num_det_terms + k ** 2 * p), dtype=float)
        cols_det = k * num_det_terms
        row = 0
        for j in range(p):
            for ing_ind in causing_ind:
                for ed_ind in caused_ind:
                    C[row, cols_det + ed_ind + k * ing_ind + k ** 2 * j] = 1
                    row += 1
        Cb = np.dot(C, vec(self.params.T))
        middle = np.linalg.inv(C @ self.cov_params() @ C.T)
        lam_wald = statistic = Cb @ middle @ Cb
        if kind.lower() == 'wald':
            df = num_restr
            dist = stats.chi2(df)
        elif kind.lower() == 'f':
            statistic = lam_wald / num_restr
            df = (num_restr, k * self.df_resid)
            dist = stats.f(*df)
        else:
            raise ValueError('kind %s not recognized' % kind)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)
        return CausalityTestResults(causing, caused, statistic, crit_value, pvalue, df, signif, test='granger', method=kind)

    def test_inst_causality(self, causing, signif=0.05):
        """
        Test for instantaneous causality

        Parameters
        ----------
        causing :
            If int or str, test whether the corresponding variable is causing
            the variable(s) specified in caused.
            If sequence of int or str, test whether the corresponding
            variables are causing the variable(s) specified in caused.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level
        verbose : bool
            If True, print a table with the results.

        Returns
        -------
        results : dict
            A dict holding the test's results. The dict's keys are:

            "statistic" : float
              The calculated test statistic.

            "crit_value" : float
              The critical value of the Chi^2-distribution.

            "pvalue" : float
              The p-value corresponding to the test statistic.

            "df" : float
              The degrees of freedom of the Chi^2-distribution.

            "conclusion" : str {"reject", "fail to reject"}
              Whether H0 can be rejected or not.

            "signif" : float
              Significance level

        Notes
        -----
        Test for instantaneous causality as described in chapters 3.6.3 and
        7.6.4 of [1]_.
        Test H0: "No instantaneous causality between caused and causing"
        against H1: "Instantaneous causality between caused and causing
        exists".

        Instantaneous causality is a symmetric relation (i.e. if causing is
        "instantaneously causing" caused, then also caused is "instantaneously
        causing" causing), thus the naming of the parameters (which is chosen
        to be in accordance with test_granger_causality()) may be misleading.

        This method is not returning the same result as JMulTi. This is
        because the test is based on a VAR(k_ar) model in statsmodels
        (in accordance to pp. 104, 320-321 in [1]_) whereas JMulTi seems
        to be using a VAR(k_ar+1) model.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        if not 0 < signif < 1:
            raise ValueError('signif has to be between 0 and 1')
        allowed_types = (str, int)
        if isinstance(causing, allowed_types):
            causing = [causing]
        if not all((isinstance(c, allowed_types) for c in causing)):
            raise TypeError('causing has to be of type string or int (or a ' + 'a sequence of these types).')
        causing = [self.names[c] if type(c) is int else c for c in causing]
        causing_ind = [util.get_index(self.names, c) for c in causing]
        caused_ind = [i for i in range(self.neqs) if i not in causing_ind]
        caused = [self.names[c] for c in caused_ind]
        k, t, p = (self.neqs, self.nobs, self.k_ar)
        num_restr = len(causing) * len(caused)
        sigma_u = self.sigma_u
        vech_sigma_u = util.vech(sigma_u)
        sig_mask = np.zeros(sigma_u.shape)
        sig_mask[causing_ind, caused_ind] = 1
        sig_mask[caused_ind, causing_ind] = 1
        vech_sig_mask = util.vech(sig_mask)
        inds = np.nonzero(vech_sig_mask)[0]
        C = np.zeros((num_restr, len(vech_sigma_u)), dtype=float)
        for row in range(num_restr):
            C[row, inds[row]] = 1
        Cs = np.dot(C, vech_sigma_u)
        d = np.linalg.pinv(duplication_matrix(k))
        Cd = np.dot(C, d)
        middle = np.linalg.inv(Cd @ np.kron(sigma_u, sigma_u) @ Cd.T) / 2
        wald_statistic = t * (Cs.T @ middle @ Cs)
        df = num_restr
        dist = stats.chi2(df)
        pvalue = dist.sf(wald_statistic)
        crit_value = dist.ppf(1 - signif)
        return CausalityTestResults(causing, caused, wald_statistic, crit_value, pvalue, df, signif, test='inst', method='wald')

    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        """
        Residual whiteness tests using Portmanteau test

        Parameters
        ----------
        nlags : int > 0
            The number of lags tested must be larger than the number of lags
            included in the VAR model.
        signif : float, between 0 and 1
            The significance level of the test.
        adjusted : bool, default False
            Flag indicating to apply small-sample adjustments.

        Returns
        -------
        WhitenessTestResults
            The test results.

        Notes
        -----
        Test the whiteness of the residuals using the Portmanteau test as
        described in [1]_, chapter 4.4.3.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        if nlags - self.k_ar <= 0:
            raise ValueError(f'The whiteness test can only be used when nlags is larger than the number of lags included in the model ({self.k_ar}).')
        statistic = 0
        u = np.asarray(self.resid)
        acov_list = _compute_acov(u, nlags)
        cov0_inv = np.linalg.inv(acov_list[0])
        for t in range(1, nlags + 1):
            ct = acov_list[t]
            to_add = np.trace(ct.T @ cov0_inv @ ct @ cov0_inv)
            if adjusted:
                to_add /= self.nobs - t
            statistic += to_add
        statistic *= self.nobs ** 2 if adjusted else self.nobs
        df = self.neqs ** 2 * (nlags - self.k_ar)
        dist = stats.chi2(df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)
        return WhitenessTestResults(statistic, crit_value, pvalue, df, signif, nlags, adjusted)

    def plot_acorr(self, nlags=10, resid=True, linewidth=8):
        """
        Plot autocorrelation of sample (endog) or residuals

        Sample (Y) or Residual autocorrelations are plotted together with the
        standard :math:`2 / \\sqrt{T}` bounds.

        Parameters
        ----------
        nlags : int
            number of lags to display (excluding 0)
        resid : bool
            If True, then the autocorrelation of the residuals is plotted
            If False, then the autocorrelation of endog is plotted.
        linewidth : int
            width of vertical bars

        Returns
        -------
        Figure
            Figure instance containing the plot.
        """
        if resid:
            acorrs = self.resid_acorr(nlags)
        else:
            acorrs = self.sample_acorr(nlags)
        bound = 2 / np.sqrt(self.nobs)
        fig = plotting.plot_full_acorr(acorrs[1:], xlabel=np.arange(1, nlags + 1), err_bound=bound, linewidth=linewidth)
        fig.suptitle('ACF plots for residuals with $2 / \\sqrt{T}$ bounds ')
        return fig

    def test_normality(self, signif=0.05):
        """
        Test assumption of normal-distributed errors using Jarque-Bera-style
        omnibus Chi^2 test.

        Parameters
        ----------
        signif : float
            Test significance level.

        Returns
        -------
        result : NormalityTestResults

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process
        """
        return test_normality(self, signif=signif)

    @cache_readonly
    def detomega(self):
        """
        Return determinant of white noise covariance with degrees of freedom
        correction:

        .. math::

            \\hat \\Omega = \\frac{T}{T - Kp - 1} \\hat \\Omega_{\\mathrm{MLE}}
        """
        return np.linalg.det(self.sigma_u)

    @cache_readonly
    def info_criteria(self):
        """information criteria for lagorder selection"""
        nobs = self.nobs
        neqs = self.neqs
        lag_order = self.k_ar
        free_params = lag_order * neqs ** 2 + neqs * self.k_exog
        if self.df_resid:
            ld = logdet_symm(self.sigma_u_mle)
        else:
            ld = -np.inf
        aic = ld + 2.0 / nobs * free_params
        bic = ld + np.log(nobs) / nobs * free_params
        hqic = ld + 2.0 * np.log(np.log(nobs)) / nobs * free_params
        if self.df_resid:
            fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)
        else:
            fpe = np.inf
        return {'aic': aic, 'bic': bic, 'hqic': hqic, 'fpe': fpe}

    @property
    def aic(self):
        """Akaike information criterion"""
        return self.info_criteria['aic']

    @property
    def fpe(self):
        """Final Prediction Error (FPE)

        Lütkepohl p. 147, see info_criteria
        """
        return self.info_criteria['fpe']

    @property
    def hqic(self):
        """Hannan-Quinn criterion"""
        return self.info_criteria['hqic']

    @property
    def bic(self):
        """Bayesian a.k.a. Schwarz info criterion"""
        return self.info_criteria['bic']

    @cache_readonly
    def roots(self):
        """
        The roots of the VAR process are the solution to
        (I - coefs[0]*z - coefs[1]*z**2 ... - coefs[p-1]*z**k_ar) = 0.
        Note that the inverse roots are returned, and stability requires that
        the roots lie outside the unit circle.
        """
        neqs = self.neqs
        k_ar = self.k_ar
        p = neqs * k_ar
        arr = np.zeros((p, p))
        arr[:neqs, :] = np.column_stack(self.coefs)
        arr[neqs:, :-neqs] = np.eye(p - neqs)
        roots = np.linalg.eig(arr)[0] ** (-1)
        idx = np.argsort(np.abs(roots))[::-1]
        return roots[idx]