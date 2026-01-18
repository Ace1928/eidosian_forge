from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def _replot_ax(ax: Axes, freq: BaseOffset):
    data = getattr(ax, '_plot_data', None)
    ax._plot_data = []
    ax.clear()
    decorate_axes(ax, freq)
    lines = []
    labels = []
    if data is not None:
        for series, plotf, kwds in data:
            series = series.copy()
            idx = series.index.asfreq(freq, how='S')
            series.index = idx
            ax._plot_data.append((series, plotf, kwds))
            if isinstance(plotf, str):
                from pandas.plotting._matplotlib import PLOT_CLASSES
                plotf = PLOT_CLASSES[plotf]._plot
            lines.append(plotf(ax, series.index._mpl_repr(), series.values, **kwds)[0])
            labels.append(pprint_thing(series.name))
    return (lines, labels)