from __future__ import annotations
from typing import (
import warnings
from matplotlib.artist import setp
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike
import pandas as pd
import pandas.core.common as com
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def _grouped_plot_by_column(plotf, data, columns=None, by=None, numeric_only: bool=True, grid: bool=False, figsize: tuple[float, float] | None=None, ax=None, layout=None, return_type=None, **kwargs):
    grouped = data.groupby(by, observed=False)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    naxes = len(columns)
    fig, axes = create_subplots(naxes=naxes, sharex=kwargs.pop('sharex', True), sharey=kwargs.pop('sharey', True), figsize=figsize, ax=ax, layout=layout)
    _axes = flatten_axes(axes)
    xlabel, ylabel = (kwargs.pop('xlabel', None), kwargs.pop('ylabel', None))
    if kwargs.get('vert', True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by
    ax_values = []
    for i, col in enumerate(columns):
        ax = _axes[i]
        gp_col = grouped[col]
        keys, values = zip(*gp_col)
        re_plotf = plotf(keys, values, ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_title(col)
        ax_values.append(re_plotf)
        ax.grid(grid)
    result = pd.Series(ax_values, index=columns, copy=False)
    if return_type is None:
        result = axes
    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f'Boxplot grouped by {byline}')
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    return result