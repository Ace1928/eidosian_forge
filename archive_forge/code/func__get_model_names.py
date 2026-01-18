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
def _get_model_names(self, latex=False):
    names = {'trend': None, 'exog': None, 'ar': None, 'ma': None, 'seasonal_ar': None, 'seasonal_ma': None, 'reduced_ar': None, 'reduced_ma': None, 'exog_variance': None, 'measurement_variance': None, 'variance': None}
    if self._k_trend > 0:
        trend_template = 't_%d' if latex else 'trend.%d'
        names['trend'] = []
        for i in self.polynomial_trend.nonzero()[0]:
            if i == 0:
                names['trend'].append('intercept')
            elif i == 1:
                names['trend'].append('drift')
            else:
                names['trend'].append(trend_template % i)
    if self._k_exog > 0:
        names['exog'] = self.exog_names
    if self.k_ar > 0:
        ar_template = '$\\phi_%d$' if latex else 'ar.L%d'
        names['ar'] = []
        for i in self.polynomial_ar.nonzero()[0][1:]:
            names['ar'].append(ar_template % i)
    if self.k_ma > 0:
        ma_template = '$\\theta_%d$' if latex else 'ma.L%d'
        names['ma'] = []
        for i in self.polynomial_ma.nonzero()[0][1:]:
            names['ma'].append(ma_template % i)
    if self.k_seasonal_ar > 0:
        seasonal_ar_template = '$\\tilde \\phi_%d$' if latex else 'ar.S.L%d'
        names['seasonal_ar'] = []
        for i in self.polynomial_seasonal_ar.nonzero()[0][1:]:
            names['seasonal_ar'].append(seasonal_ar_template % i)
    if self.k_seasonal_ma > 0:
        seasonal_ma_template = '$\\tilde \\theta_%d$' if latex else 'ma.S.L%d'
        names['seasonal_ma'] = []
        for i in self.polynomial_seasonal_ma.nonzero()[0][1:]:
            names['seasonal_ma'].append(seasonal_ma_template % i)
    if self.k_ar > 0 or self.k_seasonal_ar > 0:
        reduced_polynomial_ar = reduced_polynomial_ar = -np.polymul(self.polynomial_ar, self.polynomial_seasonal_ar)
        ar_template = '$\\Phi_%d$' if latex else 'ar.R.L%d'
        names['reduced_ar'] = []
        for i in reduced_polynomial_ar.nonzero()[0][1:]:
            names['reduced_ar'].append(ar_template % i)
    if self.k_ma > 0 or self.k_seasonal_ma > 0:
        reduced_polynomial_ma = np.polymul(self.polynomial_ma, self.polynomial_seasonal_ma)
        ma_template = '$\\Theta_%d$' if latex else 'ma.R.L%d'
        names['reduced_ma'] = []
        for i in reduced_polynomial_ma.nonzero()[0][1:]:
            names['reduced_ma'].append(ma_template % i)
    if self.state_regression and self.time_varying_regression:
        if not self.concentrate_scale:
            exog_var_template = '$\\sigma_\\text{%s}^2$' if latex else 'var.%s'
        else:
            exog_var_template = '$\\sigma_\\text{%s}^2 / \\sigma_\\zeta^2$' if latex else 'snr.%s'
        names['exog_variance'] = [exog_var_template % exog_name for exog_name in self.exog_names]
    if self.measurement_error:
        if not self.concentrate_scale:
            meas_var_tpl = '$\\sigma_\\eta^2$' if latex else 'var.measurement_error'
        else:
            meas_var_tpl = '$\\sigma_\\eta^2 / \\sigma_\\zeta^2$' if latex else 'snr.measurement_error'
        names['measurement_variance'] = [meas_var_tpl]
    if self.state_error and (not self.concentrate_scale):
        var_tpl = '$\\sigma_\\zeta^2$' if latex else 'sigma2'
        names['variance'] = [var_tpl]
    return names