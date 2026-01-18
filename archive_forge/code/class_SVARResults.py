import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
class SVARResults(SVARProcess, VARResults):
    """
    Estimate VAR(p) process with fixed number of lags

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
        List of names of the endogenous variables in order of appearance in `endog`.
    dates

    Attributes
    ----------
    aic
    bic
    bse
    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    cov_params
    dates
    detomega
    df_model : int
    df_resid : int
    endog
    endog_lagged
    fittedvalues
    fpe
    intercept
    info_criteria
    k_ar : int
    k_trend : int
    llf
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params
    k_ar : int
        Order of VAR process
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    pvalue
    names : list
        variables names
    resid
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    sigma_u_mle
    stderr
    trenorder
    tvalues
    """
    _model_type = 'SVAR'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order, A=None, B=None, A_mask=None, B_mask=None, model=None, trend='c', names=None, dates=None):
        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates
        self.n_totobs, self.neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        k_trend = util.get_trendorder(trend)
        if k_trend > 0:
            trendorder = k_trend - 1
        else:
            trendorder = None
        self.k_trend = k_trend
        self.k_exog = k_trend
        self.trendorder = trendorder
        self.exog_names = util.make_lag_names(names, lag_order, k_trend)
        self.params = params
        self.sigma_u = sigma_u
        reshaped = self.params[self.k_trend:]
        reshaped = reshaped.reshape((lag_order, self.neqs, self.neqs))
        intercept = self.params[0]
        coefs = reshaped.swapaxes(1, 2).copy()
        self.A = A
        self.B = B
        self.A_mask = A_mask
        self.B_mask = B_mask
        super().__init__(coefs, intercept, sigma_u, A, B, names=names)

    def irf(self, periods=10, var_order=None):
        """
        Analyze structural impulse responses to shocks in system

        Parameters
        ----------
        periods : int

        Returns
        -------
        irf : IRAnalysis
        """
        A = self.A
        B = self.B
        P = np.dot(npl.inv(A), B)
        return IRAnalysis(self, P=P, periods=periods, svar=True)

    def sirf_errband_mc(self, orth=False, repl=1000, steps=10, signif=0.05, seed=None, burn=100, cum=False):
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
        neqs = self.neqs
        mean = self.mean()
        k_ar = self.k_ar
        coefs = self.coefs
        sigma_u = self.sigma_u
        intercept = self.intercept
        df_model = self.df_model
        nobs = self.nobs
        ma_coll = np.zeros((repl, steps + 1, neqs, neqs))
        A = self.A
        B = self.B
        A_mask = self.A_mask
        B_mask = self.B_mask
        A_pass = self.model.A_original
        B_pass = self.model.B_original
        s_type = self.model.svar_type
        g_list = []

        def agg(impulses):
            if cum:
                return impulses.cumsum(axis=0)
            return impulses
        opt_A = A[A_mask]
        opt_B = B[B_mask]
        for i in range(repl):
            sim = util.varsim(coefs, intercept, sigma_u, seed=seed, steps=nobs + burn)
            sim = sim[burn:]
            smod = SVAR(sim, svar_type=s_type, A=A_pass, B=B_pass)
            if i == 10:
                mean_AB = np.mean(g_list, axis=0)
                split = len(A[A_mask])
                opt_A = mean_AB[:split]
                opt_B = mean_AB[split:]
            sres = smod.fit(maxlags=k_ar, A_guess=opt_A, B_guess=opt_B)
            if i < 10:
                g_list.append(np.append(sres.A[A_mask].tolist(), sres.B[B_mask].tolist()))
            ma_coll[i] = agg(sres.svar_ma_rep(maxn=steps))
        ma_sort = np.sort(ma_coll, axis=0)
        index = (int(round(signif / 2 * repl) - 1), int(round((1 - signif / 2) * repl) - 1))
        lower = ma_sort[index[0], :, :, :]
        upper = ma_sort[index[1], :, :, :]
        return (lower, upper)