from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tools.tools import maybe_unwrap_results
from ._regressionplots_doc import (
def abline_plot(intercept=None, slope=None, horiz=None, vert=None, model_results=None, ax=None, **kwargs):
    """
    Plot a line given an intercept and slope.

    Parameters
    ----------
    intercept : float
        The intercept of the line.
    slope : float
        The slope of the line.
    horiz : float or array_like
        Data for horizontal lines on the y-axis.
    vert : array_like
        Data for verterical lines on the x-axis.
    model_results : statsmodels results instance
        Any object that has a two-value `params` attribute. Assumed that it
        is (intercept, slope).
    ax : axes, optional
        Matplotlib axes instance.
    **kwargs
        Options passed to matplotlib.pyplot.plt.

    Returns
    -------
    Figure
        The figure given by `ax.figure` or a new instance.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm

    >>> np.random.seed(12345)
    >>> X = sm.add_constant(np.random.normal(0, 20, size=30))
    >>> y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
    >>> mod = sm.OLS(y,X).fit()
    >>> fig = sm.graphics.abline_plot(model_results=mod)
    >>> ax = fig.axes[0]
    >>> ax.scatter(X[:,1], y)
    >>> ax.margins(.1)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()

    .. plot:: plots/graphics_regression_abline.py
    """
    if ax is not None:
        x = ax.get_xlim()
    else:
        x = None
    fig, ax = utils.create_mpl_ax(ax)
    if model_results:
        intercept, slope = model_results.params
        if x is None:
            x = [model_results.model.exog[:, 1].min(), model_results.model.exog[:, 1].max()]
    else:
        if not (intercept is not None and slope is not None):
            raise ValueError('specify slope and intercepty or model_results')
        if x is None:
            x = ax.get_xlim()
    data_y = [x[0] * slope + intercept, x[1] * slope + intercept]
    ax.set_xlim(x)
    from matplotlib.lines import Line2D

    class ABLine2D(Line2D):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.id_xlim_callback = None
            self.id_ylim_callback = None

        def remove(self):
            ax = self.axes
            if self.id_xlim_callback:
                ax.callbacks.disconnect(self.id_xlim_callback)
            if self.id_ylim_callback:
                ax.callbacks.disconnect(self.id_ylim_callback)
            super().remove()

        def update_datalim(self, ax):
            ax.set_autoscale_on(False)
            children = ax.get_children()
            ablines = [child for child in children if child is self]
            abline = ablines[0]
            x = ax.get_xlim()
            y = [x[0] * slope + intercept, x[1] * slope + intercept]
            abline.set_data(x, y)
            ax.figure.canvas.draw()
    line = ABLine2D(x, data_y, **kwargs)
    ax.add_line(line)
    line.id_xlim_callback = ax.callbacks.connect('xlim_changed', line.update_datalim)
    line.id_ylim_callback = ax.callbacks.connect('ylim_changed', line.update_datalim)
    if horiz:
        ax.hline(horiz)
    if vert:
        ax.vline(vert)
    return fig