from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
class UECMResults(ARDLResults):
    """
    Class to hold results from fitting an UECM model.

    Parameters
    ----------
    model : UECM
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    cov_params : ndarray
        The estimated covariance matrix of the model parameters.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.
    """
    _cache: dict[str, Any] = {}

    def _ci_wrap(self, val: np.ndarray, name: str='') -> NDArray | pd.Series | pd.DataFrame:
        if not isinstance(self.model.data, PandasData):
            return val
        ndet = self.model._blocks['deterministic'].shape[1]
        nlvl = self.model._blocks['levels'].shape[1]
        lbls = self.model.exog_names[:ndet + nlvl]
        for i in range(ndet, ndet + nlvl):
            lbl = lbls[i]
            if lbl.endswith('.L1'):
                lbls[i] = lbl[:-3]
        if val.ndim == 2:
            return pd.DataFrame(val, columns=lbls, index=lbls)
        return pd.Series(val, index=lbls, name=name)

    @cache_readonly
    def ci_params(self) -> np.ndarray | pd.Series:
        """Parameters of normalized cointegrating relationship"""
        ndet = self.model._blocks['deterministic'].shape[1]
        nlvl = self.model._blocks['levels'].shape[1]
        base = np.asarray(self.params)[ndet]
        return self._ci_wrap(self.params[:ndet + nlvl] / base, 'ci_params')

    @cache_readonly
    def ci_bse(self) -> np.ndarray | pd.Series:
        """Standard Errors of normalized cointegrating relationship"""
        bse = np.sqrt(np.diag(self.ci_cov_params()))
        return self._ci_wrap(bse, 'ci_bse')

    @cache_readonly
    def ci_tvalues(self) -> np.ndarray | pd.Series:
        """T-values of normalized cointegrating relationship"""
        ndet = self.model._blocks['deterministic'].shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tvalues = np.asarray(self.ci_params) / np.asarray(self.ci_bse)
            tvalues[ndet] = np.nan
        return self._ci_wrap(tvalues, 'ci_tvalues')

    @cache_readonly
    def ci_pvalues(self) -> np.ndarray | pd.Series:
        """P-values of normalized cointegrating relationship"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.ci_tvalues)))
        return self._ci_wrap(pvalues, 'ci_pvalues')

    def ci_conf_int(self, alpha: float=0.05) -> Float64Array | pd.DataFrame:
        alpha = float_like(alpha, 'alpha')
        if self.use_t:
            q = stats.t(self.df_resid).ppf(1 - alpha / 2)
        else:
            q = stats.norm().ppf(1 - alpha / 2)
        p = self.ci_params
        se = self.ci_bse
        out = [p - q * se, p + q * se]
        if not isinstance(p, pd.Series):
            return np.column_stack(out)
        df = pd.concat(out, axis=1)
        df.columns = ['lower', 'upper']
        return df

    def ci_summary(self, alpha: float=0.05) -> Summary:

        def _ci(alpha=alpha):
            return np.asarray(self.ci_conf_int(alpha))
        smry = Summary()
        ndet = self.model._blocks['deterministic'].shape[1]
        nlvl = self.model._blocks['levels'].shape[1]
        exog_names = list(self.model.exog_names)[:ndet + nlvl]
        model = SimpleNamespace(endog_names=self.model.endog_names, exog_names=exog_names)
        data = SimpleNamespace(params=self.ci_params, bse=self.ci_bse, tvalues=self.ci_tvalues, pvalues=self.ci_pvalues, conf_int=_ci, model=model)
        tab = summary_params(data)
        tab.title = 'Cointegrating Vector'
        smry.tables.append(tab)
        return smry

    @cache_readonly
    def ci_resids(self) -> np.ndarray | pd.Series:
        d = self.model._blocks['deterministic']
        exog = self.model.data.orig_exog
        is_pandas = isinstance(exog, pd.DataFrame)
        exog = exog if is_pandas else self.model.exog
        cols = [np.asarray(d), self.model.endog]
        for key, value in self.model.dl_lags.items():
            if value is not None:
                if is_pandas:
                    cols.append(np.asarray(exog[key]))
                else:
                    cols.append(exog[:, key])
        ci_x = np.column_stack(cols)
        resids = ci_x @ self.ci_params
        if not isinstance(self.model.data, PandasData):
            return resids
        index = self.model.data.orig_endog.index
        return pd.Series(resids, index=index, name='ci_resids')

    def ci_cov_params(self) -> Float64Array | pd.DataFrame:
        """Covariance of normalized of cointegrating relationship"""
        ndet = self.model._blocks['deterministic'].shape[1]
        nlvl = self.model._blocks['levels'].shape[1]
        loc = list(range(ndet + nlvl))
        cov = self.cov_params()
        cov_a = np.asarray(cov)
        ci_cov = cov_a[np.ix_(loc, loc)]
        m = ci_cov.shape[0]
        params = np.asarray(self.params)[:ndet + nlvl]
        base = params[ndet]
        d = np.zeros((m, m))
        for i in range(m):
            if i == ndet:
                continue
            d[i, i] = 1 / base
            d[i, ndet] = -params[i] / base ** 2
        ci_cov = d @ ci_cov @ d.T
        return self._ci_wrap(ci_cov)

    def _lag_repr(self):
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""

    def bounds_test(self, case: Literal[1, 2, 3, 4, 5], cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True, asymptotic: bool=True, nsim: int=100000, seed: int | Sequence[int] | np.random.RandomState | np.random.Generator | None=None):
        """
        Cointegration bounds test of Pesaran, Shin, and Smith

        Parameters
        ----------
        case : {1, 2, 3, 4, 5}
            One of the cases covered in the PSS test.
        cov_type : str
            The covariance estimator to use. The asymptotic distribution of
            the PSS test has only been established in the homoskedastic case,
            which is the default.

            The most common choices are listed below.  Supports all covariance
            estimators that are available in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlags` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that small-sample corrections should be applied
            to the covariance estimator.
        asymptotic : bool
            Flag indicating whether to use asymptotic critical values which
            were computed by simulation (True, default) or to simulate a
            sample-size specific set of critical values. Tables are only
            available for up to 10 components in the cointegrating
            relationship, so if more variables are included then simulation
            is always used. The simulation computed the test statistic under
            and assumption that the residuals are homoskedastic.
        nsim : int
            Number of simulations to run when computing exact critical values.
            Only used if ``asymptotic`` is ``True``.
        seed : {None, int, sequence[int], RandomState, Generator}, optional
            Seed to use when simulating critical values. Must be provided if
            reproducible critical value and p-values are required when
            ``asymptotic`` is ``False``.

        Returns
        -------
        BoundsTestResult
            Named tuple containing ``stat``, ``crit_vals``, ``p_values``,
            ``null` and ``alternative``. The statistic is the F-type
            test statistic favored in PSS.

        Notes
        -----
        The PSS bounds test has 5 cases which test the coefficients on the
        level terms in the model

        .. math::

           \\Delta Y_{t}=\\delta_{0} + \\delta_{1}t + Z_{t-1}\\beta
                        + \\sum_{j=0}^{P}\\Delta X_{t-j}\\Gamma + \\epsilon_{t}

        where :math:`Z_{t-1}` contains both :math:`Y_{t-1}` and
        :math:`X_{t-1}`.

        The cases determine which deterministic terms are included in the
        model and which are tested as part of the test.

        Cases:

        1. No deterministic terms
        2. Constant included in both the model and the test
        3. Constant included in the model but not in the test
        4. Constant and trend included in the model, only trend included in
           the test
        5. Constant and trend included in the model, neither included in the
           test

        The test statistic is a Wald-type quadratic form test that all of the
        coefficients in :math:`\\beta` are 0 along with any included
        deterministic terms, which depends on the case. The statistic returned
        is an F-type test statistic which is the standard quadratic form test
        statistic divided by the number of restrictions.

        References
        ----------
        .. [*] Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing
           approaches to the analysis of level relationships. Journal of
           applied econometrics, 16(3), 289-326.
        """
        model = self.model
        trend: Literal['n', 'c', 'ct']
        if case == 1:
            trend = 'n'
        elif case in (2, 3):
            trend = 'c'
        else:
            trend = 'ct'
        order = {key: max(val) for key, val in model._order.items()}
        uecm = UECM(model.data.endog, max(model.ar_lags), model.data.orig_exog, order=order, causal=model.causal, trend=trend)
        res = uecm.fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        cov = res.cov_params()
        nvar = len(res.model.ardl_order)
        if case == 1:
            rest = np.arange(nvar)
        elif case == 2:
            rest = np.arange(nvar + 1)
        elif case == 3:
            rest = np.arange(1, nvar + 1)
        elif case == 4:
            rest = np.arange(1, nvar + 2)
        elif case == 5:
            rest = np.arange(2, nvar + 2)
        r = np.zeros((rest.shape[0], cov.shape[1]))
        for i, loc in enumerate(rest):
            r[i, loc] = 1
        vcv = r @ cov @ r.T
        coef = r @ res.params
        stat = coef.T @ np.linalg.inv(vcv) @ coef / r.shape[0]
        k = nvar
        if asymptotic and k <= 10:
            cv = pss_critical_values.crit_vals
            key = (k, case)
            upper = cv[key + (True,)]
            lower = cv[key + (False,)]
            crit_vals = pd.DataFrame({'lower': lower, 'upper': upper}, index=pss_critical_values.crit_percentiles)
            crit_vals.index.name = 'percentile'
            p_values = pd.Series({'lower': _pss_pvalue(stat, k, case, False), 'upper': _pss_pvalue(stat, k, case, True)})
        else:
            nobs = res.resid.shape[0]
            crit_vals, p_values = _pss_simulate(stat, k, case, nobs=nobs, nsim=nsim, seed=seed)
        return BoundsTestResult(stat, crit_vals, p_values, 'No Cointegration', 'Possible Cointegration')