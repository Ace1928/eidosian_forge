from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
def _plot_predictions(self, predictions, start, end, alpha, in_sample, fig, figsize):
    """Shared helper for plotting predictions"""
    from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
    _import_mpl()
    fig = create_mpl_fig(fig, figsize)
    start = 0 if start is None else start
    end = self.model._index[-1] if end is None else end
    _, _, oos, _ = self.model._get_prediction_index(start, end)
    ax = fig.add_subplot(111)
    mean = predictions.predicted_mean
    if not in_sample and oos:
        if isinstance(mean, pd.Series):
            mean = mean.iloc[-oos:]
    elif not in_sample:
        raise ValueError('in_sample is False but there are noout-of-sample forecasts to plot.')
    ax.plot(mean, zorder=2)
    if oos and alpha is not None:
        ci = np.asarray(predictions.conf_int(alpha))
        lower, upper = (ci[-oos:, 0], ci[-oos:, 1])
        label = f'{1 - alpha:.0%} confidence interval'
        x = ax.get_lines()[-1].get_xdata()
        ax.fill_between(x[-oos:], lower, upper, color='gray', alpha=0.5, label=label, zorder=1)
    ax.legend(loc='best')
    return fig