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
class UECM(ARDL):
    """
    Unconstrained Error Correlation Model(UECM)

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {int, list[int]}
        The number of lags of the endogenous variable to include in the
        model. Must be at least 1.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    order : {int, sequence[int], dict}
        If int, uses lags 0, 1, ..., order  for all exog variables. If a
        dict, applies the lags series by series. If ``exog`` is anything
        other than a DataFrame, the keys are the column index of exog
        (e.g., 0, 1, ...). If a DataFrame, keys are column names.
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
    The full specification of an UECM is

    .. math ::

       \\Delta Y_t = \\delta_0 + \\delta_1 t + \\delta_2 t^2
             + \\sum_{i=1}^{s-1} \\gamma_i I_{[(\\mod(t,s) + 1) = i]}
             + \\lambda_0 Y_{t-1} + \\lambda_1 X_{1,t-1} + \\ldots
             + \\lambda_{k} X_{k,t-1}
             + \\sum_{j=1}^{p-1} \\phi_j \\Delta Y_{t-j}
             + \\sum_{l=1}^k \\sum_{m=0}^{o_l-1} \\beta_{l,m} \\Delta X_{l, t-m}
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

    See Also
    --------
    statsmodels.tsa.ardl.ARDL
        Autoregressive distributed lag model estimation
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    >>> from statsmodels.tsa.api import UECM
    >>> from statsmodels.datasets import danish_data
    >>> data = danish_data.load_pandas().data
    >>> lrm = data.lrm
    >>> exog = data[["lry", "ibo", "ide"]]

    A basic model where all variables have 3 lags included

    >>> UECM(data.lrm, 3, data[["lry", "ibo", "ide"]], 3)

    A dictionary can be used to pass custom lag orders

    >>> UECM(data.lrm, [1, 3], exog, {"lry": 1, "ibo": 3, "ide": 2})

    Setting causal removes the 0-th lag from the exogenous variables

    >>> exog_lags = {"lry": 1, "ibo": 3, "ide": 2}
    >>> UECM(data.lrm, 3, exog, exog_lags, causal=True)

    When using NumPy arrays, the dictionary keys are the column index.

    >>> import numpy as np
    >>> lrma = np.asarray(lrm)
    >>> exoga = np.asarray(exog)
    >>> UECM(lrma, 3, exoga, {0: 1, 1: 3, 2: 2})
    """

    def __init__(self, endog: ArrayLike1D | ArrayLike2D, lags: int | None, exog: ArrayLike2D | None=None, order: _UECMOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='c', *, fixed: ArrayLike2D | None=None, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'drop', 'raise']='none') -> None:
        super().__init__(endog, lags, exog, order, trend=trend, fixed=fixed, seasonal=seasonal, causal=causal, hold_back=hold_back, period=period, missing=missing, deterministic=deterministic)
        self._results_class = UECMResults
        self._results_wrapper = UECMResultsWrapper

    def _check_lags(self, lags: int | Sequence[int] | None, hold_back: int | None) -> tuple[list[int], int]:
        """Check lags value conforms to requirement"""
        if not (isinstance(lags, _INT_TYPES) or lags is None):
            raise TypeError('lags must be an integer or None')
        return super()._check_lags(lags, hold_back)

    def _check_order(self, order: _ARDLOrder):
        """Check order conforms to requirement"""
        if isinstance(order, Mapping):
            for k, v in order.items():
                if not isinstance(v, _INT_TYPES) and v is not None:
                    raise TypeError('order values must be positive integers or None')
        elif not (isinstance(order, _INT_TYPES) or order is None):
            raise TypeError('order must be None, a positive integer, or a dict containing positive integers or None')
        order = super()._check_order(order)
        if not order:
            raise ValueError('Model must contain at least one exogenous variable')
        for key, val in order.items():
            if val == [0]:
                raise ValueError('All included exog variables must have a lag length >= 1')
        return order

    def _construct_variable_names(self):
        """Construct model variables names"""
        endog = self.data.orig_endog
        if isinstance(endog, pd.Series):
            y_base = endog.name or 'y'
        elif isinstance(endog, pd.DataFrame):
            y_base = endog.squeeze().name or 'y'
        else:
            y_base = 'y'
        y_name = f'D.{y_base}'
        x_names = list(self._deterministic_reg.columns)
        x_names.append(f'{y_base}.L1')
        orig_exog = self.data.orig_exog
        exog_pandas = isinstance(orig_exog, pd.DataFrame)
        dexog_names = []
        for key, val in self._order.items():
            if val is not None:
                if exog_pandas:
                    x_name = f'{key}.L1'
                else:
                    x_name = f'x{key}.L1'
                x_names.append(x_name)
                lag_base = x_name[:-1]
                for lag in val[:-1]:
                    dexog_names.append(f'D.{lag_base}{lag}')
        y_lags = max(self._lags) if self._lags else 0
        dendog_names = [f'{y_name}.L{lag}' for lag in range(1, y_lags)]
        x_names.extend(dendog_names)
        x_names.extend(dexog_names)
        x_names.extend(self._fixed_names)
        return (y_name, x_names)

    def _construct_regressors(self, hold_back: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Construct and format model regressors"""
        self._maxlag = max(self._lags) if self._lags else 0
        dendog = np.full_like(self.data.endog, np.nan)
        dendog[1:] = np.diff(self.data.endog, axis=0)
        dlag = max(0, self._maxlag - 1)
        self._endog_reg, self._endog = lagmat(dendog, dlag, original='sep')
        self._deterministic_reg = self._deterministics.in_sample()
        orig_exog = self.data.orig_exog
        exog_pandas = isinstance(orig_exog, pd.DataFrame)
        lvl = np.full_like(self.data.endog, np.nan)
        lvl[1:] = self.data.endog[:-1]
        lvls = [lvl.copy()]
        for key, val in self._order.items():
            if val is not None:
                if exog_pandas:
                    loc = orig_exog.columns.get_loc(key)
                else:
                    loc = key
                lvl[1:] = self.data.exog[:-1, loc]
                lvls.append(lvl.copy())
        self._levels = np.column_stack(lvls)
        if exog_pandas:
            dexog = orig_exog.diff()
        else:
            dexog = np.full_like(self.data.exog, np.nan)
            dexog[1:] = np.diff(orig_exog, axis=0)
        adj_order = {}
        for key, val in self._order.items():
            val = None if val is None or val == [1] else val[:-1]
            adj_order[key] = val
        self._exog = self._format_exog(dexog, adj_order)
        self._blocks = {'deterministic': self._deterministic_reg, 'levels': self._levels, 'endog': self._endog_reg, 'exog': self._exog, 'fixed': self._fixed}
        blocks = [self._endog]
        for key, val in self._blocks.items():
            if key != 'exog':
                blocks.append(np.asarray(val))
            else:
                for subval in val.values():
                    blocks.append(np.asarray(subval))
        y = blocks[0]
        reg = np.column_stack(blocks[1:])
        exog_maxlag = 0
        for val in self._order.values():
            exog_maxlag = max(exog_maxlag, max(val) if val is not None else 0)
        self._maxlag = max(self._maxlag, exog_maxlag)
        self._maxlag = max(self._maxlag, 1)
        if hold_back is None:
            self._hold_back = int(self._maxlag)
        if self._hold_back < self._maxlag:
            raise ValueError('hold_back must be >= the maximum lag of the endog and exog variables')
        reg = reg[self._hold_back:]
        if reg.shape[1] > reg.shape[0]:
            raise ValueError(f'The number of regressors ({reg.shape[1]}) including deterministics, lags of the endog, lags of the exogenous, and fixed regressors is larger than the sample available for estimation ({reg.shape[0]}).')
        return (np.squeeze(y)[self._hold_back:], reg)

    @Appender(str(fit_doc))
    def fit(self, *, cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True) -> UECMResults:
        params, cov_params, norm_cov_params = self._fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        res = UECMResults(self, params, cov_params, norm_cov_params, use_t=use_t)
        return UECMResultsWrapper(res)

    @classmethod
    def from_ardl(cls, ardl: ARDL, missing: Literal['none', 'drop', 'raise']='none'):
        """
        Construct a UECM from an ARDL model

        Parameters
        ----------
        ardl : ARDL
            The ARDL model instance
        missing : {"none", "drop", "raise"}, default "none"
            How to treat missing observations.

        Returns
        -------
        UECM
            The UECM model instance

        Notes
        -----
        The lag requirements for a UECM are stricter than for an ARDL.
        Any variable that is included in the UECM must have a lag length
        of at least 1. Additionally, the included lags must be contiguous
        starting at 0 if non-causal or 1 if causal.
        """
        err = 'UECM can only be created from ARDL models that include all {var_typ} lags up to the maximum lag in the model.'
        uecm_lags = {}
        dl_lags = ardl.dl_lags
        for key, val in dl_lags.items():
            max_val = max(val)
            if len(dl_lags[key]) < max_val + int(not ardl.causal):
                raise ValueError(err.format(var_typ='exogenous'))
            uecm_lags[key] = max_val
        if ardl.ar_lags is None:
            ar_lags = None
        else:
            max_val = max(ardl.ar_lags)
            if len(ardl.ar_lags) != max_val:
                raise ValueError(err.format(var_typ='endogenous'))
            ar_lags = max_val
        return cls(ardl.data.orig_endog, ar_lags, ardl.data.orig_exog, uecm_lags, trend=ardl.trend, fixed=ardl.fixed, seasonal=ardl.seasonal, hold_back=ardl.hold_back, period=ardl.period, causal=ardl.causal, missing=missing, deterministic=ardl.deterministic)

    def predict(self, params: ArrayLike1D, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None) -> np.ndarray:
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
        if dynamic is not False:
            raise NotImplementedError('dynamic forecasts are not supported')
        params, exog, exog_oos, start, end, num_oos = self._prepare_prediction(params, exog, exog_oos, start, end)
        if num_oos != 0:
            raise NotImplementedError('Out-of-sample forecasts are not supported')
        pred = np.full(self.endog.shape[0], np.nan)
        pred[-self._x.shape[0]:] = self._x @ params
        return pred[start:end + 1]

    @classmethod
    @Appender(from_formula_doc.__str__().replace('ARDL', 'UECM'))
    def from_formula(cls, formula: str, data: pd.DataFrame, lags: int | Sequence[int] | None=0, order: _ARDLOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='n', *, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'raise']='none') -> UECM:
        return super().from_formula(formula, data, lags, order, trend, causal=causal, seasonal=seasonal, deterministic=deterministic, hold_back=hold_back, period=period, missing=missing)