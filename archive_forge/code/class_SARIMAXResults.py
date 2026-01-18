from warnings import warn
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat
from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
class SARIMAXResults(MLEResults):
    """
    Class to hold results from fitting an SARIMAX model.

    Parameters
    ----------
    model : SARIMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the SARIMAX model instance.
    polynomial_ar : ndarray
        Array containing autoregressive lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_ma : ndarray
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_seasonal_ar : ndarray
        Array containing seasonal autoregressive lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_seasonal_ma : ndarray
        Array containing seasonal moving average lag polynomial coefficients,
        ordered from lowest degree to highest. Initialized with ones, unless
        a coefficient is constrained to be zero (in which case it is zero).
    polynomial_trend : ndarray
        Array containing trend polynomial coefficients, ordered from lowest
        degree to highest. Initialized with ones, unless a coefficient is
        constrained to be zero (in which case it is zero).
    model_orders : list of int
        The orders of each of the polynomials in the model.
    param_terms : list of str
        List of parameters actually included in the model, in sorted order.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)
        self.df_resid = np.inf
        self._init_kwds = self.model._get_init_kwds()
        self.specification = Bunch(**{'seasonal_periods': self.model.seasonal_periods, 'measurement_error': self.model.measurement_error, 'time_varying_regression': self.model.time_varying_regression, 'simple_differencing': self.model.simple_differencing, 'enforce_stationarity': self.model.enforce_stationarity, 'enforce_invertibility': self.model.enforce_invertibility, 'hamilton_representation': self.model.hamilton_representation, 'concentrate_scale': self.model.concentrate_scale, 'trend_offset': self.model.trend_offset, 'order': self.model.order, 'seasonal_order': self.model.seasonal_order, 'k_diff': self.model.k_diff, 'k_seasonal_diff': self.model.k_seasonal_diff, 'k_ar': self.model.k_ar, 'k_ma': self.model.k_ma, 'k_seasonal_ar': self.model.k_seasonal_ar, 'k_seasonal_ma': self.model.k_seasonal_ma, 'k_ar_params': self.model.k_ar_params, 'k_ma_params': self.model.k_ma_params, 'trend': self.model.trend, 'k_trend': self.model.k_trend, 'k_exog': self.model.k_exog, 'mle_regression': self.model.mle_regression, 'state_regression': self.model.state_regression})
        self.polynomial_trend = self.model._polynomial_trend
        self.polynomial_ar = self.model._polynomial_ar
        self.polynomial_ma = self.model._polynomial_ma
        self.polynomial_seasonal_ar = self.model._polynomial_seasonal_ar
        self.polynomial_seasonal_ma = self.model._polynomial_seasonal_ma
        self.polynomial_reduced_ar = np.polymul(self.polynomial_ar, self.polynomial_seasonal_ar)
        self.polynomial_reduced_ma = np.polymul(self.polynomial_ma, self.polynomial_seasonal_ma)
        self.model_orders = self.model.model_orders
        self.param_terms = self.model.param_terms
        start = end = 0
        for name in self.param_terms:
            if name == 'ar':
                k = self.model.k_ar_params
            elif name == 'ma':
                k = self.model.k_ma_params
            elif name == 'seasonal_ar':
                k = self.model.k_seasonal_ar_params
            elif name == 'seasonal_ma':
                k = self.model.k_seasonal_ma_params
            else:
                k = self.model_orders[name]
            end += k
            setattr(self, '_params_%s' % name, self.params[start:end])
            start += k
        all_terms = ['ar', 'ma', 'seasonal_ar', 'seasonal_ma', 'variance']
        for name in set(all_terms).difference(self.param_terms):
            setattr(self, '_params_%s' % name, np.empty(0))
        self._data_attr_model.extend(['orig_endog', 'orig_exog'])

    def extend(self, endog, exog=None, **kwargs):
        kwargs.setdefault('trend_offset', self.nobs + 1)
        return super().extend(endog, exog=exog, **kwargs)

    @cache_readonly
    def arroots(self):
        """
        (array) Roots of the reduced form autoregressive lag polynomial
        """
        return np.roots(self.polynomial_reduced_ar) ** (-1)

    @cache_readonly
    def maroots(self):
        """
        (array) Roots of the reduced form moving average lag polynomial
        """
        return np.roots(self.polynomial_reduced_ma) ** (-1)

    @cache_readonly
    def arfreq(self):
        """
        (array) Frequency of the roots of the reduced form autoregressive
        lag polynomial
        """
        z = self.arroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def mafreq(self):
        """
        (array) Frequency of the roots of the reduced form moving average
        lag polynomial
        """
        z = self.maroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def arparams(self):
        """
        (array) Autoregressive parameters actually estimated in the model.
        Does not include seasonal autoregressive parameters (see
        `seasonalarparams`) or parameters whose values are constrained to be
        zero.
        """
        return self._params_ar

    @cache_readonly
    def seasonalarparams(self):
        """
        (array) Seasonal autoregressive parameters actually estimated in the
        model. Does not include nonseasonal autoregressive parameters (see
        `arparams`) or parameters whose values are constrained to be zero.
        """
        return self._params_seasonal_ar

    @cache_readonly
    def maparams(self):
        """
        (array) Moving average parameters actually estimated in the model.
        Does not include seasonal moving average parameters (see
        `seasonalmaparams`) or parameters whose values are constrained to be
        zero.
        """
        return self._params_ma

    @cache_readonly
    def seasonalmaparams(self):
        """
        (array) Seasonal moving average parameters actually estimated in the
        model. Does not include nonseasonal moving average parameters (see
        `maparams`) or parameters whose values are constrained to be zero.
        """
        return self._params_seasonal_ma

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=0.05, start=None):
        order = ''
        if self.model.k_ar + self.model.k_diff + self.model.k_ma > 0:
            if self.model.k_ar == self.model.k_ar_params:
                order_ar = self.model.k_ar
            else:
                order_ar = list(self.model._spec.ar_lags)
            if self.model.k_ma == self.model.k_ma_params:
                order_ma = self.model.k_ma
            else:
                order_ma = list(self.model._spec.ma_lags)
            k_diff = 0 if self.model.simple_differencing else self.model.k_diff
            order = '(%s, %d, %s)' % (order_ar, k_diff, order_ma)
        seasonal_order = ''
        has_seasonal = self.model.k_seasonal_ar + self.model.k_seasonal_diff + self.model.k_seasonal_ma > 0
        if has_seasonal:
            tmp = int(self.model.k_seasonal_ar / self.model.seasonal_periods)
            if tmp == self.model.k_seasonal_ar_params:
                order_seasonal_ar = int(self.model.k_seasonal_ar / self.model.seasonal_periods)
            else:
                order_seasonal_ar = list(self.model._spec.seasonal_ar_lags)
            tmp = int(self.model.k_seasonal_ma / self.model.seasonal_periods)
            if tmp == self.model.k_ma_params:
                order_seasonal_ma = tmp
            else:
                order_seasonal_ma = list(self.model._spec.seasonal_ma_lags)
            k_seasonal_diff = self.model.k_seasonal_diff
            if self.model.simple_differencing:
                k_seasonal_diff = 0
            seasonal_order = '(%s, %d, %s, %d)' % (str(order_seasonal_ar), k_seasonal_diff, str(order_seasonal_ma), self.model.seasonal_periods)
            if not order == '':
                order += 'x'
        model_name = f'{self.model.__class__.__name__}{order}{seasonal_order}'
        return super().summary(alpha=alpha, start=start, title='SARIMAX Results', model_name=model_name)