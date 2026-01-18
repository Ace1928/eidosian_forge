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
class ARDL(AutoReg):
    """
    Autoregressive Distributed Lag (ARDL) Model

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    order : {int, sequence[int], dict}
        If int, uses lags 0, 1, ..., order  for all exog variables. If
        sequence[int], uses the ``order`` for all variables. If a dict,
        applies the lags series by series. If ``exog`` is anything other
        than a DataFrame, the keys are the column index of exog (e.g., 0,
        1, ...). If a DataFrame, keys are column names.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.

    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    The full specification of an ARDL is

    .. math ::

       Y_t = \\delta_0 + \\delta_1 t + \\delta_2 t^2
             + \\sum_{i=1}^{s-1} \\gamma_i I_{[(\\mod(t,s) + 1) = i]}
             + \\sum_{j=1}^p \\phi_j Y_{t-j}
             + \\sum_{l=1}^k \\sum_{m=0}^{o_l} \\beta_{l,m} X_{l, t-m}
             + Z_t \\lambda
             + \\epsilon_t

    where :math:`\\delta_\\bullet` capture trends, :math:`\\gamma_\\bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`\\epsilon_t` is a white noise shock. If ``causal`` is ``True``,
    then the 0-th lag of the exogenous variables is not included and the
    sum starts at ``m=1``.

    See the notebook `Autoregressive Distributed Lag Models
    <../examples/notebooks/generated/autoregressive_distributed_lag.html>`__
    for an overview.

    See Also
    --------
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.ardl.UECM
        Unconstrained Error Correction Model estimation
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    >>> from statsmodels.tsa.api import ARDL
    >>> from statsmodels.datasets import danish_data
    >>> data = danish_data.load_pandas().data
    >>> lrm = data.lrm
    >>> exog = data[["lry", "ibo", "ide"]]

    A basic model where all variables have 3 lags included

    >>> ARDL(data.lrm, 3, data[["lry", "ibo", "ide"]], 3)

    A dictionary can be used to pass custom lag orders

    >>> ARDL(data.lrm, [1, 3], exog, {"lry": 1, "ibo": 3, "ide": 2})

    Setting causal removes the 0-th lag from the exogenous variables

    >>> exog_lags = {"lry": 1, "ibo": 3, "ide": 2}
    >>> ARDL(data.lrm, [1, 3], exog, exog_lags, causal=True)

    A dictionary can also be used to pass specific lags to include.
    Sequences hold the specific lags to include, while integers are expanded
    to include [0, 1, ..., lag]. If causal is False, then the 0-th lag is
    excluded.

    >>> ARDL(lrm, [1, 3], exog, {"lry": [0, 1], "ibo": [0, 1, 3], "ide": 2})

    When using NumPy arrays, the dictionary keys are the column index.

    >>> import numpy as np
    >>> lrma = np.asarray(lrm)
    >>> exoga = np.asarray(exog)
    >>> ARDL(lrma, 3, exoga, {0: [0, 1], 1: [0, 1, 3], 2: 2})
    """

    def __init__(self, endog: Sequence[float] | pd.Series | ArrayLike2D, lags: int | Sequence[int] | None, exog: ArrayLike2D | None=None, order: _ARDLOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='c', *, fixed: ArrayLike2D | None=None, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'drop', 'raise']='none') -> None:
        self._x = np.empty((0, 0))
        self._y = np.empty((0,))
        super().__init__(endog, lags, trend=trend, seasonal=seasonal, exog=exog, hold_back=hold_back, period=period, missing=missing, deterministic=deterministic, old_names=False)
        self._causal = bool_like(causal, 'causal', strict=True)
        self.data.orig_fixed = fixed
        if fixed is not None:
            fixed_arr = array_like(fixed, 'fixed', ndim=2, maxdim=2)
            if fixed_arr.shape[0] != self.data.endog.shape[0] or not np.all(np.isfinite(fixed_arr)):
                raise ValueError('fixed must be an (nobs, m) array where nobs matches the number of observations in the endog variable, and allvalues must be finite')
            if isinstance(fixed, pd.DataFrame):
                self._fixed_names = list(fixed.columns)
            else:
                self._fixed_names = [f'z.{i}' for i in range(fixed_arr.shape[1])]
            self._fixed = fixed_arr
        else:
            self._fixed = np.empty((self.data.endog.shape[0], 0))
            self._fixed_names = []
        self._blocks: dict[str, np.ndarray] = {}
        self._names: dict[str, Sequence[str]] = {}
        self._order = self._check_order(order)
        self._y, self._x = self._construct_regressors(hold_back)
        self._endog_name, self._exog_names = self._construct_variable_names()
        self.data.param_names = self.data.xnames = self._exog_names
        self.data.ynames = self._endog_name
        self._causal = True
        if self._order:
            min_lags = [min(val) for val in self._order.values()]
            self._causal = min(min_lags) > 0
        self._results_class = ARDLResults
        self._results_wrapper = ARDLResultsWrapper

    @property
    def fixed(self) -> NDArray | pd.DataFrame | None:
        """The fixed data used to construct the model"""
        return self.data.orig_fixed

    @property
    def causal(self) -> bool:
        """Flag indicating that the ARDL is causal"""
        return self._causal

    @property
    def ar_lags(self) -> list[int] | None:
        """The autoregressive lags included in the model"""
        return None if not self._lags else self._lags

    @property
    def dl_lags(self) -> dict[Hashable, list[int]]:
        """The lags of exogenous variables included in the model"""
        return self._order

    @property
    def ardl_order(self) -> tuple[int, ...]:
        """The order of the ARDL(p,q)"""
        ar_order = 0 if not self._lags else int(max(self._lags))
        ardl_order = [ar_order]
        for lags in self._order.values():
            if lags is not None:
                ardl_order.append(int(max(lags)))
        return tuple(ardl_order)

    def _setup_regressors(self) -> None:
        """Place holder to let AutoReg init complete"""
        self._y = np.empty((self.endog.shape[0] - self._hold_back, 0))

    @staticmethod
    def _format_exog(exog: ArrayLike2D, order: dict[Hashable, list[int]]) -> dict[Hashable, np.ndarray]:
        """Transform exogenous variables and orders to regressors"""
        if not order:
            return {}
        max_order = 0
        for val in order.values():
            if val is not None:
                max_order = max(max(val), max_order)
        if not isinstance(exog, pd.DataFrame):
            exog = array_like(exog, 'exog', ndim=2, maxdim=2)
        exog_lags = {}
        for key in order:
            if order[key] is None:
                continue
            if isinstance(exog, np.ndarray):
                assert isinstance(key, int)
                col = exog[:, key]
            else:
                col = exog[key]
            lagged_col = lagmat(col, max_order, original='in')
            lags = order[key]
            exog_lags[key] = lagged_col[:, lags]
        return exog_lags

    def _check_order(self, order: _ARDLOrder) -> dict[Hashable, list[int]]:
        """Validate and standardize the model order"""
        return _format_order(self.data.orig_exog, order, self._causal)

    def _fit(self, cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._x.shape[1] == 0:
            return (np.empty((0,)), np.empty((0, 0)), np.empty((0, 0)))
        ols_mod = OLS(self._y, self._x)
        ols_res = ols_mod.fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        cov_params = ols_res.cov_params()
        use_t = ols_res.use_t
        if cov_type == 'nonrobust' and (not use_t):
            nobs = self._y.shape[0]
            k = self._x.shape[1]
            scale = nobs / (nobs - k)
            cov_params /= scale
        return (ols_res.params, cov_params, ols_res.normalized_cov_params)

    def fit(self, *, cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True) -> ARDLResults:
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
        ARDLResults
            Estimation results.

        See Also
        --------
        statsmodels.tsa.ar_model.AutoReg
            Ordinary Least Squares estimation.
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
        params, cov_params, norm_cov_params = self._fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        res = ARDLResults(self, params, cov_params, norm_cov_params, use_t=use_t)
        return ARDLResultsWrapper(res)

    def _construct_regressors(self, hold_back: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Construct and format model regressors"""
        self._maxlag = max(self._lags) if self._lags else 0
        _endog_reg, _endog = lagmat(self.data.endog, self._maxlag, original='sep')
        assert isinstance(_endog, np.ndarray)
        assert isinstance(_endog_reg, np.ndarray)
        self._endog_reg, self._endog = (_endog_reg, _endog)
        if self._endog_reg.shape[1] != len(self._lags):
            lag_locs = [lag - 1 for lag in self._lags]
            self._endog_reg = self._endog_reg[:, lag_locs]
        orig_exog = self.data.orig_exog
        self._exog = self._format_exog(orig_exog, self._order)
        exog_maxlag = 0
        for val in self._order.values():
            exog_maxlag = max(exog_maxlag, max(val) if val is not None else 0)
        self._maxlag = max(self._maxlag, exog_maxlag)
        self._deterministic_reg = self._deterministics.in_sample()
        self._blocks = {'endog': self._endog_reg, 'exog': self._exog, 'deterministic': self._deterministic_reg, 'fixed': self._fixed}
        x = [self._deterministic_reg, self._endog_reg]
        x += [ex for ex in self._exog.values()] + [self._fixed]
        reg = np.column_stack(x)
        if hold_back is None:
            self._hold_back = int(self._maxlag)
        if self._hold_back < self._maxlag:
            raise ValueError('hold_back must be >= the maximum lag of the endog and exog variables')
        reg = reg[self._hold_back:]
        if reg.shape[1] > reg.shape[0]:
            raise ValueError(f'The number of regressors ({reg.shape[1]}) including deterministics, lags of the endog, lags of the exogenous, and fixed regressors is larger than the sample available for estimation ({reg.shape[0]}).')
        return (self.data.endog[self._hold_back:], reg)

    def _construct_variable_names(self):
        """Construct model variables names"""
        y_name = self.data.ynames
        endog_lag_names = [f'{y_name}.L{i}' for i in self._lags]
        exog = self.data.orig_exog
        exog_names = {}
        for key in self._order:
            if isinstance(exog, np.ndarray):
                base = f'x{key}'
            else:
                base = str(key)
            lags = self._order[key]
            exog_names[key] = [f'{base}.L{lag}' for lag in lags]
        self._names = {'endog': endog_lag_names, 'exog': exog_names, 'deterministic': self._deterministic_reg.columns, 'fixed': self._fixed_names}
        x_names = list(self._deterministic_reg.columns)
        x_names += endog_lag_names
        for key in exog_names:
            x_names += exog_names[key]
        x_names += self._fixed_names
        return (y_name, x_names)

    def _forecasting_x(self, start: int, end: int, num_oos: int, exog: ArrayLike2D | None, exog_oos: ArrayLike2D | None, fixed: ArrayLike2D | None, fixed_oos: ArrayLike2D | None) -> np.ndarray:
        """Construct exog matrix for forecasts"""

        def pad_x(x: np.ndarray, pad: int) -> np.ndarray:
            if pad == 0:
                return x
            k = x.shape[1]
            return np.vstack([np.full((pad, k), np.nan), x])
        pad = 0 if start >= self._hold_back else self._hold_back - start
        if end + 1 < self.endog.shape[0] and exog is None and (fixed is None):
            adjusted_start = max(start - self._hold_back, 0)
            return pad_x(self._x[adjusted_start:end + 1 - self._hold_back], pad)
        exog = self.data.exog if exog is None else np.asarray(exog)
        if exog_oos is not None:
            exog = np.vstack([exog, np.asarray(exog_oos)[:num_oos]])
        fixed = self._fixed if fixed is None else np.asarray(fixed)
        if fixed_oos is not None:
            fixed = np.vstack([fixed, np.asarray(fixed_oos)[:num_oos]])
        det = self._deterministics.in_sample()
        if num_oos:
            oos_det = self._deterministics.out_of_sample(num_oos)
            det = pd.concat([det, oos_det], axis=0)
        endog = self.data.endog
        if num_oos:
            endog = np.hstack([endog, np.full(num_oos, np.nan)])
        x = [det]
        if self._lags:
            endog_reg = lagmat(endog, max(self._lags), original='ex')
            x.append(endog_reg[:, [lag - 1 for lag in self._lags]])
        if self.ardl_order[1:]:
            if isinstance(self.data.orig_exog, pd.DataFrame):
                exog = pd.DataFrame(exog, columns=self.data.orig_exog.columns)
            exog = self._format_exog(exog, self._order)
            x.extend([np.asarray(arr) for arr in exog.values()])
        if fixed.shape[1] > 0:
            x.append(fixed)
        _x = np.column_stack(x)
        _x[:self._hold_back] = np.nan
        return _x[start:]

    def predict(self, params: ArrayLike1D, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None):
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
            An array containing out-of-sample values of the exogenous
            variables. Must have the same number of columns as the exog
            used when the model was created, and at least as many rows as
            the number of out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        params, exog, exog_oos, start, end, num_oos = self._prepare_prediction(params, exog, exog_oos, start, end)

        def check_exog(arr, name, orig, exact):
            if isinstance(orig, pd.DataFrame):
                if not isinstance(arr, pd.DataFrame):
                    raise TypeError(f'{name} must be a DataFrame when the original exog was a DataFrame')
                if sorted(arr.columns) != sorted(self.data.orig_exog.columns):
                    raise ValueError(f'{name} must have the same columns as the original exog')
            else:
                arr = array_like(arr, name, ndim=2, optional=False)
            if arr.ndim != 2 or arr.shape[1] != orig.shape[1]:
                raise ValueError(f'{name} must have the same number of columns as the original data, {orig.shape[1]}')
            if exact and arr.shape[0] != orig.shape[0]:
                raise ValueError(f'{name} must have the same number of rows as the original data ({n}).')
            return arr
        n = self.data.endog.shape[0]
        if exog is not None:
            exog = check_exog(exog, 'exog', self.data.orig_exog, True)
        if exog_oos is not None:
            exog_oos = check_exog(exog_oos, 'exog_oos', self.data.orig_exog, False)
        if fixed is not None:
            fixed = check_exog(fixed, 'fixed', self._fixed, True)
        if fixed_oos is not None:
            fixed_oos = check_exog(np.asarray(fixed_oos), 'fixed_oos', self._fixed, False)
        if self._fixed.shape[1] or not self._causal:
            max_1step = 0
        else:
            max_1step = np.inf if not self._lags else min(self._lags)
            if self._order:
                min_exog = min([min(v) for v in self._order.values()])
                max_1step = min(max_1step, min_exog)
        if num_oos > max_1step:
            if self._order and exog_oos is None:
                raise ValueError('exog_oos must be provided when out-of-sample observations require values of the exog not in the original sample')
            elif self._order and exog_oos.shape[0] + max_1step < num_oos:
                raise ValueError(f'exog_oos must have at least {num_oos - max_1step} observations to produce {num_oos} forecasts based on the model specification.')
            if self._fixed.shape[1] and fixed_oos is None:
                raise ValueError('fixed_oos must be provided when predicting out-of-sample observations')
            elif self._fixed.shape[1] and fixed_oos.shape[0] < num_oos:
                raise ValueError(f'fixed_oos must have at least {num_oos} observations to produce {num_oos} forecasts.')
        if self.exog is not None and exog_oos is None and num_oos:
            exog_oos = np.full((num_oos, self.exog.shape[1]), np.nan)
            if isinstance(self.data.orig_exog, pd.DataFrame):
                exog_oos = pd.DataFrame(exog_oos, columns=self.data.orig_exog.columns)
        x = self._forecasting_x(start, end, num_oos, exog, exog_oos, fixed, fixed_oos)
        if dynamic is False:
            dynamic_start = end + 1 - start
        else:
            dynamic_step = self._parse_dynamic(dynamic, start)
            dynamic_start = dynamic_step
            if start < self._hold_back:
                dynamic_start = max(dynamic_start, self._hold_back - start)
        fcasts = np.full(x.shape[0], np.nan)
        fcasts[:dynamic_start] = x[:dynamic_start] @ params
        offset = self._deterministic_reg.shape[1]
        for i in range(dynamic_start, fcasts.shape[0]):
            for j, lag in enumerate(self._lags):
                loc = i - lag
                if loc >= dynamic_start:
                    val = fcasts[loc]
                else:
                    val = self.endog[start + loc]
                x[i, offset + j] = val
            fcasts[i] = x[i] @ params
        return self._wrap_prediction(fcasts, start, end + 1 + num_oos, 0)

    @classmethod
    def from_formula(cls, formula: str, data: pd.DataFrame, lags: int | Sequence[int] | None=0, order: _ARDLOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='n', *, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'raise']='none') -> ARDL | UECM:
        """
        Construct an ARDL from a formula

        Parameters
        ----------
        formula : str
            Formula with form dependent ~ independent | fixed. See Examples
            below.
        data : DataFrame
            DataFrame containing the variables in the formula.
        lags : {int, list[int]}
            The number of lags to include in the model if an integer or the
            list of lag indices to include.  For example, [1, 4] will only
            include lags 1 and 4 while lags=4 will include lags 1, 2, 3,
            and 4.
        order : {int, sequence[int], dict}
            If int, uses lags 0, 1, ..., order  for all exog variables. If
            sequence[int], uses the ``order`` for all variables. If a dict,
            applies the lags series by series. If ``exog`` is anything other
            than a DataFrame, the keys are the column index of exog (e.g., 0,
            1, ...). If a DataFrame, keys are column names.
        causal : bool, optional
            Whether to include lag 0 of exog variables.  If True, only
            includes lags 1, 2, ...
        trend : {'n', 'c', 't', 'ct'}, optional
            The trend to include in the model:

            * 'n' - No trend.
            * 'c' - Constant only.
            * 't' - Time trend only.
            * 'ct' - Constant and time trend.

            The default is 'c'.

        seasonal : bool, optional
            Flag indicating whether to include seasonal dummies in the model.
            If seasonal is True and trend includes 'c', then the first period
            is excluded from the seasonal terms.
        deterministic : DeterministicProcess, optional
            A deterministic process.  If provided, trend and seasonal are
            ignored. A warning is raised if trend is not "n" and seasonal
            is not False.
        hold_back : {None, int}, optional
            Initial observations to exclude from the estimation sample.  If
            None, then hold_back is equal to the maximum lag in the model.
            Set to a non-zero value to produce comparable models with
            different lag length.  For example, to compare the fit of a model
            with lags=3 and lags=1, set hold_back=3 which ensures that both
            models are estimated using observations 3,...,nobs. hold_back
            must be >= the maximum lag in the model.
        period : {None, int}, optional
            The period of the data. Only used if seasonal is True. This
            parameter can be omitted if using a pandas object for endog
            that contains a recognized frequency.
        missing : {"none", "drop", "raise"}, optional
            Available options are 'none', 'drop', and 'raise'. If 'none', no
            NaN checking is done. If 'drop', any observations with NaNs are
            dropped. If 'raise', an error is raised. Default is 'none'.

        Returns
        -------
        ARDL
            The ARDL model instance

        Examples
        --------
        A simple ARDL using the Danish data

        >>> from statsmodels.datasets.danish_data import load
        >>> from statsmodels.tsa.api import ARDL
        >>> data = load().data
        >>> mod = ARDL.from_formula("lrm ~ ibo", data, 2, 2)

        Fixed regressors can be specified using a |

        >>> mod = ARDL.from_formula("lrm ~ ibo | ide", data, 2, 2)
        """
        index = data.index
        fixed_formula = None
        if '|' in formula:
            formula, fixed_formula = formula.split('|')
            fixed_formula = fixed_formula.strip()
        mod = OLS.from_formula(formula + ' -1', data)
        exog = mod.data.orig_exog
        exog.index = index
        endog = mod.data.orig_endog
        endog.index = index
        if fixed_formula is not None:
            endog_name = formula.split('~')[0].strip()
            fixed_formula = f'{endog_name} ~ {fixed_formula} - 1'
            mod = OLS.from_formula(fixed_formula, data)
            fixed: pd.DataFrame | None = mod.data.orig_exog
            fixed.index = index
        else:
            fixed = None
        return cls(endog, lags, exog, order, trend=trend, fixed=fixed, causal=causal, seasonal=seasonal, deterministic=deterministic, hold_back=hold_back, period=period, missing=missing)