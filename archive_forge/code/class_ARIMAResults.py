from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE
from statsmodels.tsa.statespace.tools import diff
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.gls import gls as estimate_gls
from statsmodels.tsa.arima.specification import SARIMAXSpecification
@Appender(sarimax.SARIMAXResults.__doc__)
class ARIMAResults(sarimax.SARIMAXResults):

    @Appender(sarimax.SARIMAXResults.append.__doc__)
    def append(self, endog, exog=None, refit=False, fit_kwargs=None, **kwargs):
        if exog is not None:
            orig_exog = self.model.data.orig_exog
            exog_names = self.model.exog_names
            self.model.data.orig_exog = self.model._input_exog
            self.model.exog_names = self.model._input_exog_names
        out = super().append(endog, exog=exog, refit=refit, fit_kwargs=fit_kwargs, **kwargs)
        if exog is not None:
            self.model.data.orig_exog = orig_exog
            self.model.exog_names = exog_names
        return out