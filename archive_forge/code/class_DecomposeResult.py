import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period
class DecomposeResult:
    """
    Results class for seasonal decompositions

    Parameters
    ----------
    observed : array_like
        The data series that has been decomposed.
    seasonal : array_like
        The seasonal component of the data series.
    trend : array_like
        The trend component of the data series.
    resid : array_like
        The residual component of the data series.
    weights : array_like, optional
        The weights used to reduce outlier influence.
    """

    def __init__(self, observed, seasonal, trend, resid, weights=None):
        self._seasonal = seasonal
        self._trend = trend
        if weights is None:
            weights = np.ones_like(observed)
            if isinstance(observed, pd.Series):
                weights = pd.Series(weights, index=observed.index, name='weights')
        self._weights = weights
        self._resid = resid
        self._observed = observed

    @property
    def observed(self):
        """Observed data"""
        return self._observed

    @property
    def seasonal(self):
        """The estimated seasonal component"""
        return self._seasonal

    @property
    def trend(self):
        """The estimated trend component"""
        return self._trend

    @property
    def resid(self):
        """The estimated residuals"""
        return self._resid

    @property
    def weights(self):
        """The weights used in the robust estimation"""
        return self._weights

    @property
    def nobs(self):
        """Number of observations"""
        return self._observed.shape

    def plot(self, observed=True, seasonal=True, trend=True, resid=True, weights=False):
        """
        Plot estimated components

        Parameters
        ----------
        observed : bool
            Include the observed series in the plot
        seasonal : bool
            Include the seasonal component in the plot
        trend : bool
            Include the trend component in the plot
        resid : bool
            Include the residual in the plot
        weights : bool
            Include the weights in the plot (if any)

        Returns
        -------
        matplotlib.figure.Figure
            The figure instance that containing the plot.
        """
        from pandas.plotting import register_matplotlib_converters
        from statsmodels.graphics.utils import _import_mpl
        plt = _import_mpl()
        register_matplotlib_converters()
        series = [(self._observed, 'Observed')] if observed else []
        series += [(self.trend, 'trend')] if trend else []
        if self.seasonal.ndim == 1:
            series += [(self.seasonal, 'seasonal')] if seasonal else []
        elif self.seasonal.ndim > 1:
            if isinstance(self.seasonal, pd.DataFrame):
                for col in self.seasonal.columns:
                    series += [(self.seasonal[col], 'seasonal')] if seasonal else []
            else:
                for i in range(self.seasonal.shape[1]):
                    series += [(self.seasonal[:, i], 'seasonal')] if seasonal else []
        series += [(self.resid, 'residual')] if resid else []
        series += [(self.weights, 'weights')] if weights else []
        if isinstance(self._observed, (pd.DataFrame, pd.Series)):
            nobs = self._observed.shape[0]
            xlim = (self._observed.index[0], self._observed.index[nobs - 1])
        else:
            xlim = (0, self._observed.shape[0] - 1)
        fig, axs = plt.subplots(len(series), 1, sharex=True)
        for i, (ax, (series, def_name)) in enumerate(zip(axs, series)):
            if def_name != 'residual':
                ax.plot(series)
            else:
                ax.plot(series, marker='o', linestyle='none')
                ax.plot(xlim, (0, 0), color='#000000', zorder=-3)
            name = getattr(series, 'name', def_name)
            if def_name != 'Observed':
                name = name.capitalize()
            title = ax.set_title if i == 0 and observed else ax.set_ylabel
            title(name)
            ax.set_xlim(xlim)
        fig.tight_layout()
        return fig