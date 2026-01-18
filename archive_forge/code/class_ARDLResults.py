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
class ARDLResults(AutoRegResults):
    """
    Class to hold results from fitting an ARDL model.

    Parameters
    ----------
    model : ARDL
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
    use_t : bool
        Whether use_t was set in fit
    """
    _cache = {}

    def __init__(self, model: ARDL, params: np.ndarray, cov_params: np.ndarray, normalized_cov_params: Float64Array | None=None, scale: float=1.0, use_t: bool=False):
        super().__init__(model, params, normalized_cov_params, scale, use_t=use_t)
        self._cache = {}
        self._params = params
        self._nobs = model.nobs
        self._n_totobs = model.endog.shape[0]
        self._df_model = model.df_model
        self._ar_lags = model.ar_lags
        self._max_lag = 0
        if self._ar_lags:
            self._max_lag = max(self._ar_lags)
        self._hold_back = self.model.hold_back
        self.cov_params_default = cov_params

    @Appender(remove_parameters(ARDL.predict.__doc__, 'params'))
    def predict(self, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None):
        return self.model.predict(self._params, start=start, end=end, dynamic=dynamic, exog=exog, exog_oos=exog_oos, fixed=fixed, fixed_oos=fixed_oos)

    def forecast(self, steps: int=1, exog: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None) -> np.ndarray | pd.Series:
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : {int, str, datetime}, default 1
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency,
            steps must be an integer.
        exog : array_like, optional
            Exogenous values to use out-of-sample. Must have same number of
            columns as original exog data and at least `steps` rows
        fixed : array_like, optional
            Fixed values to use out-of-sample. Must have same number of
            columns as original fixed data and at least `steps` rows

        Returns
        -------
        array_like
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.

        See Also
        --------
        ARDLResults.predict
            In- and out-of-sample predictions
        ARDLResults.get_prediction
            In- and out-of-sample predictions and confidence intervals
        """
        start = self.model.data.orig_endog.shape[0]
        if isinstance(steps, (int, np.integer)):
            end = start + steps - 1
        else:
            end = steps
        return self.predict(start=start, end=end, dynamic=False, exog_oos=exog, fixed_oos=fixed)

    def _lag_repr(self) -> np.ndarray:
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        ar_lags = self._ar_lags if self._ar_lags is not None else []
        k_ar = len(ar_lags)
        ar_params = np.zeros(self._max_lag + 1)
        ar_params[0] = 1
        offset = self.model._deterministic_reg.shape[1]
        params = self._params[offset:offset + k_ar]
        for i, lag in enumerate(ar_lags):
            ar_params[lag] = -params[i]
        return ar_params

    def get_prediction(self, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None) -> np.ndarray | pd.Series:
        """
        Predictions and prediction intervals

        Parameters
        ----------
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
        PredictionResults
            Prediction results with mean and prediction intervals
        """
        mean = self.predict(start=start, end=end, dynamic=dynamic, exog=exog, exog_oos=exog_oos, fixed=fixed, fixed_oos=fixed_oos)
        mean_var = np.full_like(mean, fill_value=self.sigma2)
        mean_var[np.isnan(mean)] = np.nan
        start = 0 if start is None else start
        end = self.model._index[-1] if end is None else end
        _, _, oos, _ = self.model._get_prediction_index(start, end)
        if oos > 0:
            ar_params = self._lag_repr()
            ma = arma2ma(ar_params, np.ones(1), lags=oos)
            mean_var[-oos:] = self.sigma2 * np.cumsum(ma ** 2)
        if isinstance(mean, pd.Series):
            mean_var = pd.Series(mean_var, index=mean.index)
        return PredictionResults(mean, mean_var)

    @Substitution(predict_params=_predict_params)
    def plot_predict(self, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None, alpha: float=0.05, in_sample: bool=True, fig: matplotlib.figure.Figure=None, figsize: tuple[int, int] | None=None) -> matplotlib.figure.Figure:
        """
        Plot in- and out-of-sample predictions

        Parameters
        ----------
%(predict_params)s
        alpha : {float, None}
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float]
            Tuple containing the figure size values.

        Returns
        -------
        Figure
            Figure handle containing the plot.
        """
        predictions = self.get_prediction(start=start, end=end, dynamic=dynamic, exog=exog, exog_oos=exog_oos, fixed=fixed, fixed_oos=fixed_oos)
        return self._plot_predictions(predictions, start, end, alpha, in_sample, fig, figsize)

    def summary(self, alpha: float=0.05) -> Summary:
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        Summary
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        model = self.model
        title = model.__class__.__name__ + ' Model Results'
        method = 'Conditional MLE'
        start = self._hold_back
        if self.data.dates is not None:
            dates = self.data.dates
            sample = [dates[start].strftime('%m-%d-%Y')]
            sample += ['- ' + dates[-1].strftime('%m-%d-%Y')]
        else:
            sample = [str(start), str(len(self.data.orig_endog))]
        model = self.model.__class__.__name__ + str(self.model.ardl_order)
        if self.model.seasonal:
            model = 'Seas. ' + model
        dep_name = str(self.model.endog_names)
        top_left = [('Dep. Variable:', [dep_name]), ('Model:', [model]), ('Method:', [method]), ('Date:', None), ('Time:', None), ('Sample:', [sample[0]]), ('', [sample[1]])]
        top_right = [('No. Observations:', [str(len(self.model.endog))]), ('Log Likelihood', ['%#5.3f' % self.llf]), ('S.D. of innovations', ['%#5.3f' % self.sigma2 ** 0.5]), ('AIC', ['%#5.3f' % self.aic]), ('BIC', ['%#5.3f' % self.bic]), ('HQIC', ['%#5.3f' % self.hqic])]
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, title=title)
        smry.add_table_params(self, alpha=alpha, use_t=False)
        return smry