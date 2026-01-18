from __future__ import annotations
from abc import (
from collections.abc import (
from typing import (
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
from pandas.plotting._matplotlib.tools import (
class AreaPlot(LinePlot):

    @property
    def _kind(self) -> Literal['area']:
        return 'area'

    def __init__(self, data, **kwargs) -> None:
        kwargs.setdefault('stacked', True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Downcasting object dtype arrays', category=FutureWarning)
            data = data.fillna(value=0)
        LinePlot.__init__(self, data, **kwargs)
        if not self.stacked:
            self.kwds.setdefault('alpha', 0.5)
        if self.logy or self.loglog:
            raise ValueError('Log-y scales are not supported in area plot')

    @classmethod
    def _plot(cls, ax: Axes, x, y: np.ndarray, style=None, column_num=None, stacking_id=None, is_errorbar: bool=False, **kwds):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds['label'])
        line_kwds = kwds.copy()
        line_kwds.pop('label')
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)
        xdata, y_values = lines[0].get_data(orig=False)
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            start = ax._stacker_pos_prior[stacking_id]
        elif (y <= 0).all():
            start = ax._stacker_neg_prior[stacking_id]
        else:
            start = np.zeros(len(y))
        if 'color' not in kwds:
            kwds['color'] = lines[0].get_color()
        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        res = [rect]
        return res

    def _post_plot_logic(self, ax: Axes, data) -> None:
        LinePlot._post_plot_logic(self, ax, data)
        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        if self.ylim is None and (not is_shared_y):
            if (data >= 0).all().all():
                ax.set_ylim(0, None)
            elif (data <= 0).all().all():
                ax.set_ylim(None, 0)