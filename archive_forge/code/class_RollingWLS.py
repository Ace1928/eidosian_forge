from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
from statsmodels.tools.validation import array_like, int_like, string_like
@Substitution(model_type='Weighted', model='WLS', parameters=common_params, extra_parameters=extra_parameters)
@Appender(_doc)
class RollingWLS:

    def __init__(self, endog, exog, window=None, *, weights=None, min_nobs=None, missing='drop', expanding=False):
        missing = string_like(missing, 'missing', options=('drop', 'raise', 'skip'))
        temp_msng = 'drop' if missing != 'raise' else 'raise'
        Model.__init__(self, endog, exog, missing=temp_msng, hasconst=None)
        k_const = self.k_constant
        const_idx = self.data.const_idx
        Model.__init__(self, endog, exog, missing='none', hasconst=False)
        self.k_constant = k_const
        self.data.const_idx = const_idx
        self._y = array_like(endog, 'endog')
        nobs = self._y.shape[0]
        self._x = array_like(exog, 'endog', ndim=2, shape=(nobs, None))
        window = int_like(window, 'window', optional=True)
        weights = array_like(weights, 'weights', optional=True, shape=(nobs,))
        self._window = window if window is not None else self._y.shape[0]
        self._weighted = weights is not None
        self._weights = np.ones(nobs) if weights is None else weights
        w12 = np.sqrt(self._weights)
        self._wy = w12 * self._y
        self._wx = w12[:, None] * self._x
        min_nobs = int_like(min_nobs, 'min_nobs', optional=True)
        self._min_nobs = min_nobs if min_nobs is not None else self._x.shape[1]
        if self._min_nobs < self._x.shape[1] or self._min_nobs > self._window:
            raise ValueError('min_nobs must be larger than the number of regressors in the model and less than window')
        self._expanding = expanding
        self._is_nan = np.zeros_like(self._y, dtype=bool)
        self._has_nan = self._find_nans()
        self.const_idx = self.data.const_idx
        self._skip_missing = missing == 'skip'

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        return Model._handle_data(self, endog, exog, missing, hasconst, **kwargs)

    def _find_nans(self):
        nans = np.isnan(self._y)
        nans |= np.any(np.isnan(self._x), axis=1)
        nans |= np.isnan(self._weights)
        self._is_nan[:] = nans
        has_nan = np.cumsum(nans)
        w = self._window
        has_nan[w - 1:] = has_nan[w - 1:] - has_nan[:-(w - 1)]
        if self._expanding:
            has_nan[:self._min_nobs] = False
        else:
            has_nan[:w - 1] = False
        return has_nan.astype(bool)

    def _get_data(self, idx):
        window = self._window
        if idx >= window:
            loc = slice(idx - window, idx)
        else:
            loc = slice(idx)
        y = self._y[loc]
        wy = self._wy[loc]
        wx = self._wx[loc]
        weights = self._weights[loc]
        missing = self._is_nan[loc]
        not_missing = ~missing
        if np.any(missing):
            y = y[not_missing]
            wy = wy[not_missing]
            wx = wx[not_missing]
            weights = weights[not_missing]
        return (y, wy, wx, weights, not_missing)

    def _fit_single(self, idx, wxpwx, wxpwy, nobs, store, params_only, method):
        if nobs < self._min_nobs:
            return
        try:
            if method == 'inv' or not params_only:
                wxpwxi = np.linalg.inv(wxpwx)
            if method == 'inv':
                params = wxpwxi @ wxpwy
            else:
                _, wy, wx, _, _ = self._get_data(idx)
                if method == 'lstsq':
                    params = lstsq(wx, wy)[0]
                else:
                    wxpwxiwxp = np.linalg.pinv(wx)
                    params = wxpwxiwxp @ wy
        except np.linalg.LinAlgError:
            return
        store.params[idx - 1] = params
        if params_only:
            return
        y, wy, wx, weights, _ = self._get_data(idx)
        wresid, ssr, llf = self._loglike(params, wy, wx, weights, nobs)
        wxwresid = wx * wresid[:, None]
        wxepwxe = wxwresid.T @ wxwresid
        tot_params = wx.shape[1]
        s2 = ssr / (nobs - tot_params)
        centered_tss, uncentered_tss = self._sum_of_squares(y, wy, weights)
        store.ssr[idx - 1] = ssr
        store.llf[idx - 1] = llf
        store.nobs[idx - 1] = nobs
        store.s2[idx - 1] = s2
        store.xpxi[idx - 1] = wxpwxi
        store.xeex[idx - 1] = wxepwxe
        store.centered_tss[idx - 1] = centered_tss
        store.uncentered_tss[idx - 1] = uncentered_tss

    def _loglike(self, params, wy, wx, weights, nobs):
        nobs2 = nobs / 2.0
        wresid = wy - wx @ params
        ssr = np.sum(wresid ** 2, axis=0)
        llf = -np.log(ssr) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        llf += 0.5 * np.sum(np.log(weights))
        return (wresid, ssr, llf)

    def _sum_of_squares(self, y, wy, weights):
        mean = np.average(y, weights=weights)
        centered_tss = np.sum(weights * (y - mean) ** 2)
        uncentered_tss = np.dot(wy, wy)
        return (centered_tss, uncentered_tss)

    def _reset(self, idx):
        """Compute xpx and xpy using a single dot product"""
        _, wy, wx, _, not_missing = self._get_data(idx)
        nobs = not_missing.sum()
        xpx = wx.T @ wx
        xpy = wx.T @ wy
        return (xpx, xpy, nobs)

    def fit(self, method='inv', cov_type='nonrobust', cov_kwds=None, reset=None, use_t=False, params_only=False):
        """
        Estimate model parameters.

        Parameters
        ----------
        method : {'inv', 'lstsq', 'pinv'}
            Method to use when computing the the model parameters.

            * 'inv' - use moving windows inner-products and matrix inversion.
              This method is the fastest, but may be less accurate than the
              other methods.
            * 'lstsq' - Use numpy.linalg.lstsq
            * 'pinv' - Use numpy.linalg.pinv. This method matches the default
              estimator in non-moving regression estimators.
        cov_type : {'nonrobust', 'HCCM', 'HC0'}
            Covariance estimator:

            * nonrobust - The classic OLS covariance estimator
            * HCCM, HC0 - White heteroskedasticity robust covariance
        cov_kwds : dict
            Unused
        reset : int, optional
            Interval to recompute the moving window inner products used to
            estimate the model parameters. Smaller values improve accuracy,
            although in practice this setting is not required to be set.
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.
        params_only : bool, optional
            Flag indicating that only parameters should be computed. Avoids
            calculating all other statistics or performing inference.

        Returns
        -------
        RollingRegressionResults
            Estimation results where all pre-sample values are nan-filled.
        """
        method = string_like(method, 'method', options=('inv', 'lstsq', 'pinv'))
        reset = int_like(reset, 'reset', optional=True)
        reset = self._y.shape[0] if reset is None else reset
        if reset < 1:
            raise ValueError('reset must be a positive integer')
        nobs, k = self._x.shape
        store = RollingStore(params=np.full((nobs, k), np.nan), ssr=np.full(nobs, np.nan), llf=np.full(nobs, np.nan), nobs=np.zeros(nobs, dtype=int), s2=np.full(nobs, np.nan), xpxi=np.full((nobs, k, k), np.nan), xeex=np.full((nobs, k, k), np.nan), centered_tss=np.full(nobs, np.nan), uncentered_tss=np.full(nobs, np.nan))
        w = self._window
        first = self._min_nobs if self._expanding else w
        xpx, xpy, nobs = self._reset(first)
        if not (self._has_nan[first - 1] and self._skip_missing):
            self._fit_single(first, xpx, xpy, nobs, store, params_only, method)
        wx, wy = (self._wx, self._wy)
        for i in range(first + 1, self._x.shape[0] + 1):
            if self._has_nan[i - 1] and self._skip_missing:
                continue
            if i % reset == 0:
                xpx, xpy, nobs = self._reset(i)
            else:
                if not self._is_nan[i - w - 1] and i > w:
                    remove_x = wx[i - w - 1:i - w]
                    xpx -= remove_x.T @ remove_x
                    xpy -= remove_x.T @ wy[i - w - 1:i - w]
                    nobs -= 1
                if not self._is_nan[i - 1]:
                    add_x = wx[i - 1:i]
                    xpx += add_x.T @ add_x
                    xpy += add_x.T @ wy[i - 1:i]
                    nobs += 1
            self._fit_single(i, xpx, xpy, nobs, store, params_only, method)
        return RollingRegressionResults(self, store, self.k_constant, use_t, cov_type)

    @classmethod
    @Appender(Model.from_formula.__doc__)
    def from_formula(cls, formula, data, window, weights=None, subset=None, *args, **kwargs):
        if subset is not None:
            data = data.loc[subset]
        eval_env = kwargs.pop('eval_env', None)
        if eval_env is None:
            eval_env = 2
        elif eval_env == -1:
            from patsy import EvalEnvironment
            eval_env = EvalEnvironment({})
        else:
            eval_env += 1
        missing = kwargs.get('missing', 'skip')
        from patsy import NAAction, dmatrices
        na_action = NAAction(on_NA='raise', NA_types=[])
        result = dmatrices(formula, data, eval_env, return_type='dataframe', NA_action=na_action)
        endog, exog = result
        if endog.ndim > 1 and endog.shape[1] > 1 or endog.ndim > 2:
            raise ValueError('endog has evaluated to an array with multiple columns that has shape {}. This occurs when the variable converted to endog is non-numeric (e.g., bool or str).'.format(endog.shape))
        kwargs.update({'missing': missing, 'window': window})
        if weights is not None:
            kwargs['weights'] = weights
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula
        mod.data.frame = data
        return mod