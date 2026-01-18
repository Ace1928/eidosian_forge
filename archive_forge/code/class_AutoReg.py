from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
class AutoReg(tsa_model.TimeSeriesModel):
    """
    Autoregressive AR-X(p) model

    Estimate an AR-X model using Conditional Maximum Likelihood (OLS).

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {None, int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
        None excludes all AR lags, and behave identically to 0.
    trend : {'n', 'c', 't', 'ct'}
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

    seasonal : bool
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    exog : array_like, optional
        Exogenous variables to include in the model. Must have the same number
        of observations as endog and should be aligned so that endog[i] is
        regressed on exog[i].
    hold_back : {None, int}
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    deterministic : DeterministicProcess
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    old_names : bool
        Flag indicating whether to use the v0.11 names or the v0.12+ names.

        .. deprecated:: 0.13.0

           old_names is deprecated and will be removed after 0.14 is
           released. You must update any code reliant on the old variable
           names to use the new names.

    See Also
    --------
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Estimation of SARIMAX models using exact likelihood and the
        Kalman Filter.

    Notes
    -----
    See the notebook `Autoregressions
    <../examples/notebooks/generated/autoregressions.html>`__ for an overview.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.ar_model import AutoReg
    >>> data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']
    >>> out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'

    Start by fitting an unrestricted Seasonal AR model

    >>> res = AutoReg(data, lags = [1, 11, 12]).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 5.945, HQIC: 5.970, BIC: 6.007

    An alternative used seasonal dummies

    >>> res = AutoReg(data, lags=1, seasonal=True, period=11).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 6.017, HQIC: 6.080, BIC: 6.175

    Finally, both the seasonal AR structure and dummies can be included

    >>> res = AutoReg(data, lags=[1, 11, 12], seasonal=True, period=11).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 5.884, HQIC: 5.959, BIC: 6.071
    """
    _y: Float64Array

    def __init__(self, endog: ArrayLike1D, lags: int | Sequence[int] | None, trend: Literal['n', 'c', 't', 'ct']='c', seasonal: bool=False, exog: ArrayLike2D | None=None, hold_back: int | None=None, period: int | None=None, missing: str='none', *, deterministic: DeterministicProcess | None=None, old_names: bool=False):
        super().__init__(endog, exog, None, None, missing=missing)
        self._trend = cast(Literal['n', 'c', 't', 'ct'], string_like(trend, 'trend', options=('n', 'c', 't', 'ct'), optional=False))
        self._seasonal = bool_like(seasonal, 'seasonal')
        self._period = int_like(period, 'period', optional=True)
        if self._period is None and self._seasonal:
            self._period = _get_period(self.data, self._index_freq)
        terms: list[DeterministicTerm] = [TimeTrend.from_string(self._trend)]
        if seasonal:
            assert isinstance(self._period, int)
            terms.append(Seasonality(self._period))
        if hasattr(self.data.orig_endog, 'index'):
            index = self.data.orig_endog.index
        else:
            index = np.arange(self.data.endog.shape[0])
        self._user_deterministic = False
        if deterministic is not None:
            if not isinstance(deterministic, DeterministicProcess):
                raise TypeError('deterministic must be a DeterministicProcess')
            self._deterministics = deterministic
            self._user_deterministic = True
        else:
            self._deterministics = DeterministicProcess(index, additional_terms=terms)
        self._exog_names: list[str] = []
        self._k_ar = 0
        self._old_names = bool_like(old_names, 'old_names', optional=False)
        if deterministic is not None and (self._trend != 'n' or self._seasonal):
            warnings.warn('When using deterministic, trend must be "n" and seasonal must be False.', SpecificationWarning, stacklevel=2)
        if self._old_names:
            warnings.warn('old_names will be removed after the 0.14 release. You should stop setting this parameter and use the new names.', FutureWarning, stacklevel=2)
        self._lags, self._hold_back = self._check_lags(lags, int_like(hold_back, 'hold_back', optional=True))
        self._setup_regressors()
        self.nobs = self._y.shape[0]
        self.data.xnames = self.exog_names

    @property
    def ar_lags(self) -> list[int] | None:
        """The autoregressive lags included in the model"""
        lags = list(self._lags)
        return None if not lags else lags

    @property
    def hold_back(self) -> int | None:
        """The number of initial obs. excluded from the estimation sample."""
        return self._hold_back

    @property
    def trend(self) -> Literal['n', 'c', 'ct', 'ctt']:
        """The trend used in the model."""
        return self._trend

    @property
    def seasonal(self) -> bool:
        """Flag indicating that the model contains a seasonal component."""
        return self._seasonal

    @property
    def deterministic(self) -> DeterministicProcess | None:
        """The deterministic used to construct the model"""
        return self._deterministics if self._user_deterministic else None

    @property
    def period(self) -> int | None:
        """The period of the seasonal component."""
        return self._period

    @property
    def df_model(self) -> int:
        """The model degrees of freedom."""
        return self._x.shape[1]

    @property
    def exog_names(self) -> list[str] | None:
        """Names of exogenous variables included in model"""
        return self._exog_names

    def initialize(self) -> None:
        """Initialize the model (no-op)."""
        pass

    def _check_lags(self, lags: int | Sequence[int] | None, hold_back: int | None) -> tuple[list[int], int]:
        if lags is None:
            _lags: list[int] = []
            self._maxlag = 0
        elif isinstance(lags, Iterable):
            _lags = []
            for lag in lags:
                val = int_like(lag, 'lags')
                assert isinstance(val, int)
                _lags.append(val)
            _lags_arr: NDArray = np.array(sorted(_lags))
            if np.any(_lags_arr < 1) or np.unique(_lags_arr).shape[0] != _lags_arr.shape[0]:
                raise ValueError('All values in lags must be positive and distinct.')
            self._maxlag = np.max(_lags_arr)
            _lags = [int(v) for v in _lags_arr]
        else:
            val = int_like(lags, 'lags')
            assert isinstance(val, int)
            self._maxlag = val
            if self._maxlag < 0:
                raise ValueError('lags must be a non-negative scalar.')
            _lags_arr = np.arange(1, self._maxlag + 1)
            _lags = [int(v) for v in _lags_arr]
        if hold_back is None:
            hold_back = self._maxlag
        if hold_back < self._maxlag:
            raise ValueError('hold_back must be >= lags if lags is an int ormax(lags) if lags is array_like.')
        return (_lags, int(hold_back))

    def _setup_regressors(self) -> None:
        maxlag = self._maxlag
        hold_back = self._hold_back
        exog_names = []
        endog_names = self.endog_names
        x, y = lagmat(self.endog, maxlag, original='sep')
        exog_names.extend([endog_names + f'.L{lag}' for lag in self._lags])
        if len(self._lags) < maxlag:
            x = x[:, np.asarray(self._lags) - 1]
        self._k_ar = x.shape[1]
        deterministic = self._deterministics.in_sample()
        if deterministic.shape[1]:
            x = np.c_[to_numpy(deterministic), x]
            if self._old_names:
                deterministic_names = []
                if 'c' in self._trend:
                    deterministic_names.append('intercept')
                if 't' in self._trend:
                    deterministic_names.append('trend')
                if self._seasonal:
                    period = self._period
                    assert isinstance(period, int)
                    names = [f'seasonal.{i}' for i in range(period)]
                    if 'c' in self._trend:
                        names = names[1:]
                    deterministic_names.extend(names)
            else:
                deterministic_names = list(deterministic.columns)
            exog_names = deterministic_names + exog_names
        if self.exog is not None:
            x = np.c_[x, self.exog]
            exog_names.extend(self.data.param_names)
        y = y[hold_back:]
        x = x[hold_back:]
        if y.shape[0] < x.shape[1]:
            reg = x.shape[1]
            period = self._period
            trend = 0 if self._trend == 'n' else len(self._trend)
            if self._seasonal:
                assert isinstance(period, int)
                seas = period - int('c' in self._trend)
            else:
                seas = 0
            lags = len(self._lags)
            nobs = y.shape[0]
            raise ValueError(f'The model specification cannot be estimated. The model contains {reg} regressors ({trend} trend, {seas} seasonal, {lags} lags) but after adjustment for hold_back and creation of the lags, there are only {nobs} data points available to estimate parameters.')
        self._y, self._x = (y, x)
        self._exog_names = exog_names

    def fit(self, cov_type: str='nonrobust', cov_kwds: dict[str, Any] | None=None, use_t: bool=False) -> AutoRegResultsWrapper:
        """
        Estimate the model parameters.

        Parameters
        ----------
        cov_type : str
            The covariance estimator to use. The most common choices are listed
            below.  Supports all covariance estimators that are available
            in ``OLS.fit``.

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
            A flag indicating that inference should use the Student's t
            distribution that accounts for model degree of freedom.  If False,
            uses the normal distribution. If None, defers the choice to
            the cov_type. It also removes degree of freedom corrections from
            the covariance estimator when cov_type is 'nonrobust'.

        Returns
        -------
        AutoRegResults
            Estimation results.

        See Also
        --------
        statsmodels.regression.linear_model.OLS
            Ordinary Least Squares estimation.
        statsmodels.regression.linear_model.RegressionResults
            See ``get_robustcov_results`` for a detailed list of available
            covariance estimators and options.

        Notes
        -----
        Use ``OLS`` to estimate model parameters and to estimate parameter
        covariance.
        """
        if self._x.shape[1] == 0:
            return AutoRegResultsWrapper(AutoRegResults(self, np.empty(0), np.empty((0, 0))))
        ols_mod = OLS(self._y, self._x)
        ols_res = ols_mod.fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        cov_params = ols_res.cov_params()
        use_t = ols_res.use_t
        if cov_type == 'nonrobust' and (not use_t):
            nobs = self._y.shape[0]
            k = self._x.shape[1]
            scale = nobs / (nobs - k)
            cov_params /= scale
        res = AutoRegResults(self, ols_res.params, cov_params, ols_res.normalized_cov_params, use_t=use_t)
        return AutoRegResultsWrapper(res)

    def _resid(self, params: ArrayLike) -> np.ndarray:
        params = array_like(params, 'params', ndim=2)
        return self._y.squeeze() - (self._x @ params).squeeze()

    def loglike(self, params: ArrayLike) -> float:
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Returns
        -------
        float
            The log-likelihood value.
        """
        nobs = self.nobs
        resid = self._resid(params)
        ssr = resid @ resid
        llf = -(nobs / 2) * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1)
        return llf

    def score(self, params: ArrayLike) -> np.ndarray:
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        resid = self._resid(params)
        return self._x.T @ resid

    def information(self, params: ArrayLike) -> np.ndarray:
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        ndarray
            The information matrix.
        """
        resid = self._resid(params)
        sigma2 = resid @ resid / self.nobs
        return self._x.T @ self._x * (1 / sigma2)

    def hessian(self, params: ArrayLike) -> np.ndarray:
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        return -self.information(params)

    def _setup_oos_forecast(self, add_forecasts: int, exog_oos: ArrayLike2D) -> np.ndarray:
        x = np.zeros((add_forecasts, self._x.shape[1]))
        oos_exog = self._deterministics.out_of_sample(steps=add_forecasts)
        n_deterministic = oos_exog.shape[1]
        x[:, :n_deterministic] = to_numpy(oos_exog)
        loc = n_deterministic + len(self._lags)
        if self.exog is not None:
            exog_oos_a = np.asarray(exog_oos)
            x[:, loc:] = exog_oos_a[:add_forecasts]
        return x

    def _wrap_prediction(self, prediction: np.ndarray, start: int, end: int, pad: int) -> pd.Series:
        prediction = np.hstack([np.full(pad, np.nan), prediction])
        n_values = end - start + pad
        if not isinstance(self.data.orig_endog, (pd.Series, pd.DataFrame)):
            return prediction[-n_values:]
        index = self._index
        if end > self.endog.shape[0]:
            freq = getattr(index, 'freq', None)
            if freq:
                if isinstance(index, pd.PeriodIndex):
                    index = pd.period_range(index[0], freq=freq, periods=end)
                else:
                    index = pd.date_range(index[0], freq=freq, periods=end)
            else:
                index = pd.RangeIndex(end)
        index = index[start - pad:end]
        prediction = prediction[-n_values:]
        return pd.Series(prediction, index=index)

    def _dynamic_predict(self, params: ArrayLike, start: int, end: int, dynamic: int, num_oos: int, exog: Float64Array | None, exog_oos: Float64Array | None) -> pd.Series:
        """

        :param params:
        :param start:
        :param end:
        :param dynamic:
        :param num_oos:
        :param exog:
        :param exog_oos:
        :return:
        """
        reg = []
        hold_back = self._hold_back
        adj = 0
        if start < hold_back:
            adj = hold_back - start
        start += adj
        dynamic = max(dynamic - adj, 0)
        if start - hold_back <= self.nobs:
            is_loc = slice(start - hold_back, end + 1 - hold_back)
            x = self._x[is_loc]
            if exog is not None:
                x = x.copy()
                x[:, -exog.shape[1]:] = exog[start:end + 1]
            reg.append(x)
        if num_oos > 0:
            reg.append(self._setup_oos_forecast(num_oos, exog_oos))
        _reg = np.vstack(reg)
        det_col_idx = self._x.shape[1] - len(self._lags)
        det_col_idx -= 0 if self.exog is None else self.exog.shape[1]
        forecasts = np.empty(_reg.shape[0])
        forecasts[:dynamic] = _reg[:dynamic] @ params
        for h in range(dynamic, _reg.shape[0]):
            for j, lag in enumerate(self._lags):
                fcast_loc = h - lag
                if fcast_loc >= dynamic:
                    val = forecasts[fcast_loc]
                else:
                    val = self.endog[fcast_loc + start]
                _reg[h, det_col_idx + j] = val
            forecasts[h] = np.squeeze(_reg[h:h + 1] @ params)
        return self._wrap_prediction(forecasts, start, end + 1 + num_oos, adj)

    def _static_oos_predict(self, params: ArrayLike, num_oos: int, exog_oos: ArrayLike2D) -> np.ndarray:
        new_x = self._setup_oos_forecast(num_oos, exog_oos)
        if self._maxlag == 0:
            return new_x @ params
        forecasts = np.empty(num_oos)
        nexog = 0 if self.exog is None else self.exog.shape[1]
        ar_offset = self._x.shape[1] - nexog - len(self._lags)
        for i in range(num_oos):
            for j, lag in enumerate(self._lags):
                loc = i - lag
                val = self._y[loc] if loc < 0 else forecasts[loc]
                new_x[i, ar_offset + j] = np.squeeze(val)
            forecasts[i] = np.squeeze(new_x[i:i + 1] @ params)
        return forecasts

    def _static_predict(self, params: Float64Array, start: int, end: int, num_oos: int, exog: Float64Array | None, exog_oos: Float64Array | None) -> pd.Series:
        """
        Path for static predictions

        Parameters
        ----------
        params : ndarray
            The model parameters
        start : int
            Index of first observation
        end : int
            Index of last in-sample observation. Inclusive, so start:end+1
            in slice notation.
        num_oos : int
            Number of out-of-sample observations, so that the returned size is
            num_oos + (end - start + 1).
        exog : {ndarray, DataFrame}
            Array containing replacement exog values
        exog_oos :  {ndarray, DataFrame}
            Containing forecast exog values
        """
        hold_back = self._hold_back
        nobs = self.endog.shape[0]
        x = np.empty((0, self._x.shape[1]))
        adj = max(0, hold_back - start)
        start += adj
        if start <= nobs:
            is_loc = slice(start - hold_back, end + 1 - hold_back)
            x = self._x[is_loc]
            if exog is not None:
                exog_a = np.asarray(exog)
                x = x.copy()
                x[:, -exog_a.shape[1]:] = exog_a[start:end + 1]
        in_sample = x @ params
        if num_oos == 0:
            return self._wrap_prediction(in_sample, start, end + 1, adj)
        out_of_sample = self._static_oos_predict(params, num_oos, exog_oos)
        prediction = np.hstack((in_sample, out_of_sample))
        return self._wrap_prediction(prediction, start, end + 1 + num_oos, adj)

    def _prepare_prediction(self, params: ArrayLike, exog: ArrayLike2D, exog_oos: ArrayLike2D, start: int | str | datetime.datetime | pd.Timestamp | None, end: int | str | datetime.datetime | pd.Timestamp | None) -> tuple[np.ndarray, np.ndarray | pd.DataFrame | None, np.ndarray | pd.DataFrame | None, int, int, int]:
        params = array_like(params, 'params')
        assert isinstance(params, np.ndarray)
        if isinstance(exog, pd.DataFrame):
            _exog = exog
        else:
            _exog = array_like(exog, 'exog', ndim=2, optional=True)
        if isinstance(exog_oos, pd.DataFrame):
            _exog_oos = exog_oos
        else:
            _exog_oos = array_like(exog_oos, 'exog_oos', ndim=2, optional=True)
        start = 0 if start is None else start
        end = self._index[-1] if end is None else end
        start, end, num_oos, _ = self._get_prediction_index(start, end)
        return (params, _exog, _exog_oos, start, end, num_oos)

    def _parse_dynamic(self, dynamic, start):
        if isinstance(dynamic, (str, bytes, pd.Timestamp, dt.datetime, pd.Period)):
            dynamic_loc, _, _ = self._get_index_loc(dynamic)
            dynamic_loc -= start
        elif dynamic is True:
            dynamic_loc = 0
        else:
            dynamic_loc = int(dynamic)
        if dynamic_loc < 0:
            raise ValueError('Dynamic prediction cannot begin prior to the first observation in the sample.')
        return dynamic_loc

    def predict(self, params: ArrayLike, start: int | str | datetime.datetime | pd.Timestamp | None=None, end: int | str | datetime.datetime | pd.Timestamp | None=None, dynamic: bool | int=False, exog: ArrayLike2D | None=None, exog_oos: ArrayLike2D | None=None) -> pd.Series:
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        params : array_like
            The fitted model parameters.
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous variable.
            Must has the same number of columns as the exog used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        params, exog, exog_oos, start, end, num_oos = self._prepare_prediction(params, exog, exog_oos, start, end)
        if self.exog is None and (exog is not None or exog_oos is not None):
            raise ValueError('exog and exog_oos cannot be used when the model does not contains exogenous regressors.')
        elif self.exog is not None:
            if exog is not None and exog.shape != self.exog.shape:
                msg = 'The shape of exog {0} must match the shape of the exog variable used to create the model {1}.'
                raise ValueError(msg.format(exog.shape, self.exog.shape))
            if exog_oos is not None and exog_oos.shape[1] != self.exog.shape[1]:
                msg = 'The number of columns in exog_oos ({0}) must match the number of columns  in the exog variable used to create the model ({1}).'
                raise ValueError(msg.format(exog_oos.shape[1], self.exog.shape[1]))
            if num_oos > 0 and exog_oos is None:
                raise ValueError('exog_oos must be provided when producing out-of-sample forecasts.')
            elif exog_oos is not None and num_oos > exog_oos.shape[0]:
                msg = 'start and end indicate that {0} out-of-sample predictions must be computed. exog_oos has {1} rows but must have at least {0}.'
                raise ValueError(msg.format(num_oos, exog_oos.shape[0]))
        if isinstance(dynamic, bool) and (not dynamic) or self._maxlag == 0:
            return self._static_predict(params, start, end, num_oos, exog, exog_oos)
        dynamic = self._parse_dynamic(dynamic, start)
        return self._dynamic_predict(params, start, end, dynamic, num_oos, exog, exog_oos)