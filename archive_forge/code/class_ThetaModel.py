from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.validation import (
from statsmodels.tsa.deterministic import DeterministicTerm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import add_trend, freq_to_period
class ThetaModel:
    """
    The Theta forecasting model of Assimakopoulos and Nikolopoulos (2000)

    Parameters
    ----------
    endog : array_like, 1d
        The data to forecast.
    period : int, default None
        The period of the data that is used in the seasonality test and
        adjustment. If None then the period is determined from y's index,
        if available.
    deseasonalize : bool, default True
        A flag indicating whether the deseasonalize the data. If True and
        use_test is True, the data is only deseasonalized if the null of no
        seasonal component is rejected.
    use_test : bool, default True
        A flag indicating whether test the period-th autocorrelation. If this
        test rejects using a size of 10%, then decomposition is used. Set to
        False to skip the test.
    method : {"auto", "additive", "multiplicative"}, default "auto"
        The model used for the seasonal decomposition. "auto" uses a
        multiplicative if y is non-negative and all estimated seasonal
        components are positive. If either of these conditions is False,
        then it uses an additive decomposition.
    difference : bool, default False
        A flag indicating to difference the data before testing for
        seasonality.

    See Also
    --------
    statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing
        Exponential smoothing parameter estimation and forecasting
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA parameter estimation and forecasting

    Notes
    -----
    The Theta model forecasts the future as a weighted combination of two
    Theta lines.  This class supports combinations of models with two
    thetas: 0 and a user-specified choice (default 2). The forecasts are
    then

    .. math::

       \\hat{X}_{T+h|T} = \\frac{\\theta-1}{\\theta} b_0
                         \\left[h - 1 + \\frac{1}{\\alpha}
                         - \\frac{(1-\\alpha)^T}{\\alpha} \\right]
                         + \\tilde{X}_{T+h|T}

    where :math:`\\tilde{X}_{T+h|T}` is the SES forecast of the endogenous
    variable using the parameter :math:`\\alpha`. :math:`b_0` is the
    slope of a time trend line fitted to X using the terms 0, 1, ..., T-1.

    The model is estimated in steps:

    1. Test for seasonality
    2. Deseasonalize if seasonality detected
    3. Estimate :math:`\\alpha` by fitting a SES model to the data and
       :math:`b_0` by OLS.
    4. Forecast the series
    5. Reseasonalize if the data was deseasonalized.

    The seasonality test examines where the autocorrelation at the
    seasonal period is different from zero. The seasonality is then
    removed using a seasonal decomposition with a multiplicative trend.
    If the seasonality estimate is non-positive then an additive trend
    is used instead. The default deseasonalizing method can be changed
    using the options.

    References
    ----------
    .. [1] Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a
       decomposition approach to forecasting. International Journal of
       Forecasting, 16(4), 521-530.
    .. [2] Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
       International Journal of Forecasting, 19(2), 287-290.
    .. [3] Fioruci, J. A., Pellegrini, T. R., Louzada, F., & Petropoulos, F.
       (2015). The optimized theta method. arXiv preprint arXiv:1503.03529.
    """

    def __init__(self, endog, *, period: Optional[int]=None, deseasonalize: bool=True, use_test: bool=True, method: str='auto', difference: bool=False) -> None:
        self._y = array_like(endog, 'endog', ndim=1)
        if isinstance(endog, pd.DataFrame):
            self.endog_orig = endog.iloc[:, 0]
        else:
            self.endog_orig = endog
        self._period = int_like(period, 'period', optional=True)
        self._deseasonalize = bool_like(deseasonalize, 'deseasonalize')
        self._use_test = bool_like(use_test, 'use_test') and self._deseasonalize
        self._diff = bool_like(difference, 'difference')
        self._method = string_like(method, 'model', options=('auto', 'additive', 'multiplicative', 'mul', 'add'))
        if self._method == 'auto':
            self._method = 'mul' if self._y.min() > 0 else 'add'
        if self._period is None and self._deseasonalize:
            idx = getattr(endog, 'index', None)
            pfreq = None
            if idx is not None:
                pfreq = getattr(idx, 'freq', None)
                if pfreq is None:
                    pfreq = getattr(idx, 'inferred_freq', None)
            if pfreq is not None:
                self._period = freq_to_period(pfreq)
            else:
                raise ValueError('You must specify a period or endog must be a pandas object with a DatetimeIndex with a freq not set to None')
        self._has_seasonality = self._deseasonalize

    def _test_seasonality(self) -> None:
        y = self._y
        if self._diff:
            y = np.diff(y)
        rho = acf(y, nlags=self._period, fft=True)
        nobs = y.shape[0]
        stat = nobs * rho[-1] ** 2 / np.sum(rho[:-1] ** 2)
        self._has_seasonality = stat > 2.705543454095404

    def _deseasonalize_data(self) -> tuple[np.ndarray, np.ndarray]:
        y = self._y
        if not self._has_seasonality:
            return (self._y, np.empty(0))
        res = seasonal_decompose(y, model=self._method, period=self._period)
        if res.seasonal.min() <= 0:
            self._method = 'add'
            res = seasonal_decompose(y, model='add', period=self._period)
            return (y - res.seasonal, res.seasonal[:self._period])
        else:
            return (y / res.seasonal, res.seasonal[:self._period])

    def fit(self, use_mle: bool=False, disp: bool=False) -> 'ThetaModelResults':
        """
        Estimate model parameters.

        Parameters
        ----------
        use_mle : bool, default False
            Estimate the parameters using MLE by fitting an ARIMA(0,1,1) with
            a drift.  If False (the default), estimates parameters using OLS
            of a constant and a time-trend and by fitting a SES to the model
            data.
        disp : bool, default True
            Display iterative output from fitting the model.

        Notes
        -----
        When using MLE, the parameters are estimated from the ARIMA(0,1,1)

        .. math::

           X_t = X_{t-1} + b_0 + (\\alpha-1)\\epsilon_{t-1} + \\epsilon_t

        When estimating the model using 2-step estimation, the model
        parameters are estimated using the OLS regression

        .. math::

           X_t = a_0 + b_0 (t-1) + \\eta_t

        and the SES

        .. math::

           \\tilde{X}_{t+1} = \\alpha X_{t} + (1-\\alpha)\\tilde{X}_{t}

        Returns
        -------
        ThetaModelResult
            Model results and forecasting
        """
        if self._deseasonalize and self._use_test:
            self._test_seasonality()
        y, seasonal = self._deseasonalize_data()
        if use_mle:
            mod = SARIMAX(y, order=(0, 1, 1), trend='c')
            res = mod.fit(disp=disp)
            params = np.asarray(res.params)
            alpha = params[1] + 1
            if alpha > 1:
                alpha = 0.9998
                res = mod.fit_constrained({'ma.L1': alpha - 1})
                params = np.asarray(res.params)
            b0 = params[0]
            sigma2 = params[-1]
            one_step = res.forecast(1) - b0
        else:
            ct = add_trend(y, 'ct', prepend=True)[:, :2]
            ct[:, 1] -= 1
            _, b0 = np.linalg.lstsq(ct, y, rcond=None)[0]
            res = ExponentialSmoothing(y, initial_level=y[0], initialization_method='known').fit(disp=disp)
            alpha = res.params[0]
            sigma2 = None
            one_step = res.forecast(1)
        return ThetaModelResults(b0, alpha, sigma2, one_step, seasonal, use_mle, self)

    @property
    def deseasonalize(self) -> bool:
        """Whether to deseasonalize the data"""
        return self._deseasonalize

    @property
    def period(self) -> int:
        """The period of the seasonality"""
        return self._period

    @property
    def use_test(self) -> bool:
        """Whether to test the data for seasonality"""
        return self._use_test

    @property
    def difference(self) -> bool:
        """Whether the data is differenced in the seasonality test"""
        return self._diff

    @property
    def method(self) -> str:
        """The method used to deseasonalize the data"""
        return self._method