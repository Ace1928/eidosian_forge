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
class ThetaModelResults:
    """
    Results class from estimated Theta Models.

    Parameters
    ----------
    b0 : float
        The estimated trend slope.
    alpha : float
        The estimated SES parameter.
    sigma2 : float
        The estimated residual variance from the SES/IMA model.
    one_step : float
        The one-step forecast from the SES.
    seasonal : ndarray
        An array of estimated seasonal terms.
    use_mle : bool
        A flag indicating that the parameters were estimated using MLE.
    model : ThetaModel
        The model used to produce the results.
    """

    def __init__(self, b0: float, alpha: float, sigma2: Optional[float], one_step: float, seasonal: np.ndarray, use_mle: bool, model: ThetaModel) -> None:
        self._b0 = b0
        self._alpha = alpha
        self._sigma2 = sigma2
        self._one_step = one_step
        self._nobs = model.endog_orig.shape[0]
        self._model = model
        self._seasonal = seasonal
        self._use_mle = use_mle

    @property
    def params(self) -> pd.Series:
        """The forecasting model parameters"""
        return pd.Series([self._b0, self._alpha], index=['b0', 'alpha'])

    @property
    def sigma2(self) -> float:
        """The estimated residual variance"""
        if self._sigma2 is None:
            mod = SARIMAX(self.model._y, order=(0, 1, 1), trend='c')
            res = mod.fit(disp=False)
            self._sigma2 = np.asarray(res.params)[-1]
        assert self._sigma2 is not None
        return self._sigma2

    @property
    def model(self) -> ThetaModel:
        """The model used to produce the results"""
        return self._model

    def forecast(self, steps: int=1, theta: float=2) -> pd.Series:
        """
        Forecast the model for a given theta

        Parameters
        ----------
        steps : int
            The number of steps ahead to compute the forecast components.
        theta : float
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.

        Returns
        -------
        Series
            A Series containing the forecasts

        Notes
        -----
        The forecast is computed as

        .. math::

           \\hat{X}_{T+h|T} = \\frac{\\theta-1}{\\theta} b_0
                             \\left[h - 1 + \\frac{1}{\\alpha}
                             - \\frac{(1-\\alpha)^T}{\\alpha} \\right]
                             + \\tilde{X}_{T+h|T}

        where :math:`\\tilde{X}_{T+h|T}` is the SES forecast of the endogenous
        variable using the parameter :math:`\\alpha`. :math:`b_0` is the
        slope of a time trend line fitted to X using the terms 0, 1, ..., T-1.

        This expression follows from [1]_ and [2]_ when the combination
        weights are restricted to be (theta-1)/theta and 1/theta. This nests
        the original implementation when theta=2 and the two weights are both
        1/2.

        References
        ----------
        .. [1] Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
           International Journal of Forecasting, 19(2), 287-290.
        .. [2] Fioruci, J. A., Pellegrini, T. R., Louzada, F., & Petropoulos,
           F. (2015). The optimized theta method. arXiv preprint
           arXiv:1503.03529.
        """
        steps = int_like(steps, 'steps')
        if steps < 1:
            raise ValueError('steps must be a positive integer')
        theta = float_like(theta, 'theta')
        if theta < 1:
            raise ValueError('theta must be a float >= 1')
        thresh = 4.0 / np.finfo(np.double).eps
        trend_weight = (theta - 1) / theta if theta < thresh else 1.0
        comp = self.forecast_components(steps=steps)
        fcast = trend_weight * comp.trend + np.asarray(comp.ses)
        if self.model.deseasonalize:
            seasonal = np.asarray(comp.seasonal)
            if self.model.method.startswith('mul'):
                fcast *= seasonal
            else:
                fcast += seasonal
        fcast.name = 'forecast'
        return fcast

    def forecast_components(self, steps: int=1) -> pd.DataFrame:
        """
        Compute the three components of the Theta model forecast

        Parameters
        ----------
        steps : int
            The number of steps ahead to compute the forecast components.

        Returns
        -------
        DataFrame
            A DataFrame with three columns: trend, ses and seasonal containing
            the forecast values of each of the three components.

        Notes
        -----
        For a given value of :math:`\\theta`, the deseasonalized forecast is
        `fcast = w * trend + ses` where :math:`w = \\frac{theta - 1}{theta}`.
        The reseasonalized forecasts are then `seasonal * fcast` if the
        seasonality is multiplicative or `seasonal + fcast` if the seasonality
        is additive.
        """
        steps = int_like(steps, 'steps')
        if steps < 1:
            raise ValueError('steps must be a positive integer')
        alpha = self._alpha
        b0 = self._b0
        nobs = self._nobs
        h = np.arange(1, steps + 1, dtype=np.float64) - 1
        if alpha > 0:
            h += 1 / alpha - (1 - alpha) ** nobs / alpha
        trend = b0 * h
        ses = self._one_step * np.ones(steps)
        if self.model.method.startswith('add'):
            season = np.zeros(steps)
        else:
            season = np.ones(steps)
        if self.model.deseasonalize:
            seasonal = self._seasonal
            period = self.model.period
            oos_idx = nobs + np.arange(steps)
            seasonal_locs = oos_idx % period
            if seasonal.shape[0]:
                season[:] = seasonal[seasonal_locs]
        index = getattr(self.model.endog_orig, 'index', None)
        if index is None:
            index = pd.RangeIndex(0, self.model.endog_orig.shape[0])
        index = extend_index(steps, index)
        df = pd.DataFrame({'trend': trend, 'ses': ses, 'seasonal': season}, index=index)
        return df

    def summary(self) -> Summary:
        """
        Summarize the model

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
        smry = Summary()
        model_name = type(model).__name__
        title = model_name + ' Results'
        method = 'MLE' if self._use_mle else 'OLS/SES'
        is_series = isinstance(model.endog_orig, pd.Series)
        index = getattr(model.endog_orig, 'index', None)
        if is_series and isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
            sample = [index[0].strftime('%m-%d-%Y')]
            sample += ['- ' + index[-1].strftime('%m-%d-%Y')]
        else:
            sample = [str(0), str(model.endog_orig.shape[0])]
        dep_name = getattr(model.endog_orig, 'name', 'endog') or 'endog'
        top_left = [('Dep. Variable:', [dep_name]), ('Method:', [method]), ('Date:', None), ('Time:', None), ('Sample:', [sample[0]]), ('', [sample[1]])]
        method = 'Multiplicative' if model.method.startswith('mul') else 'Additive'
        top_right = [('No. Observations:', [str(self._nobs)]), ('Deseasonalized:', [str(model.deseasonalize)])]
        if model.deseasonalize:
            top_right.extend([('Deseas. Method:', [method]), ('Period:', [str(model.period)]), ('', ['']), ('', [''])])
        else:
            top_right.extend([('', [''])] * 4)
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, title=title)
        table_fmt = {'data_fmts': ['%s', '%#0.4g'], 'data_aligns': 'r'}
        data = np.asarray(self.params)[:, None]
        st = SimpleTable(data, ['Parameters', 'Estimate'], list(self.params.index), title='Parameter Estimates', txt_fmt=table_fmt)
        smry.tables.append(st)
        return smry

    def prediction_intervals(self, steps: int=1, theta: float=2, alpha: float=0.05) -> pd.DataFrame:
        """
        Parameters
        ----------
        steps : int, default 1
            The number of steps ahead to compute the forecast components.
        theta : float, default 2
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.
        alpha : float, default 0.05
            Significance level for the confidence intervals.

        Returns
        -------
        DataFrame
            DataFrame with columns lower and upper

        Notes
        -----
        The variance of the h-step forecast is assumed to follow from the
        integrated Moving Average structure of the Theta model, and so is
        :math:`\\sigma^2(1 + (h-1)(1 + (\\alpha-1)^2)`. The prediction interval
        assumes that innovations are normally distributed.
        """
        model_alpha = self.params.iloc[1]
        sigma2_h = (1 + np.arange(steps) * (1 + (model_alpha - 1) ** 2)) * self.sigma2
        sigma_h = np.sqrt(sigma2_h)
        quantile = stats.norm.ppf(alpha / 2)
        predictions = self.forecast(steps, theta)
        return pd.DataFrame({'lower': predictions + sigma_h * quantile, 'upper': predictions + sigma_h * -quantile})

    def plot_predict(self, steps: int=1, theta: float=2, alpha: Optional[float]=0.05, in_sample: bool=False, fig: Optional['matplotlib.figure.Figure']=None, figsize: tuple[float, float]=None) -> 'matplotlib.figure.Figure':
        """
        Plot forecasts, prediction intervals and in-sample values

        Parameters
        ----------
        steps : int, default 1
            The number of steps ahead to compute the forecast components.
        theta : float, default 2
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.
        alpha : {float, None}, default 0.05
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool, default False
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure, default None
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float], default None
            Tuple containing the figure size.

        Returns
        -------
        Figure
            Figure handle containing the plot.

        Notes
        -----
        The variance of the h-step forecast is assumed to follow from the
        integrated Moving Average structure of the Theta model, and so is
        :math:`\\sigma^2(\\alpha^2 + (h-1))`. The prediction interval assumes
        that innovations are normally distributed.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        assert fig is not None
        predictions = self.forecast(steps, theta)
        pred_index = predictions.index
        ax = fig.add_subplot(111)
        nobs = self.model.endog_orig.shape[0]
        index = pd.Index(np.arange(nobs))
        if in_sample:
            if isinstance(self.model.endog_orig, pd.Series):
                index = self.model.endog_orig.index
            ax.plot(index, self.model.endog_orig)
        ax.plot(pred_index, predictions)
        if alpha is not None:
            pi = self.prediction_intervals(steps, theta, alpha)
            label = f'{1 - alpha:.0%} confidence interval'
            ax.fill_between(pred_index, pi['lower'], pi['upper'], color='gray', alpha=0.5, label=label)
        ax.legend(loc='best', frameon=False)
        fig.tight_layout(pad=1.0)
        return fig