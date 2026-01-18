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
class PiePlot(MPLPlot):

    @property
    def _kind(self) -> Literal['pie']:
        return 'pie'
    _layout_type = 'horizontal'

    def __init__(self, data, kind=None, **kwargs) -> None:
        data = data.fillna(value=0)
        if (data < 0).any().any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: bool | None | Literal['sym']) -> bool | None | Literal['sym']:
        super()._validate_log_kwd(kwd=kwd, value=value)
        if value is not False:
            warnings.warn(f"PiePlot ignores the '{kwd}' keyword", UserWarning, stacklevel=find_stack_level())
        return False

    def _validate_color_args(self, color, colormap) -> None:
        return None

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors(num_colors=len(self.data), color_kwds='colors')
        self.kwds.setdefault('colors', colors)
        for i, (label, y) in enumerate(self._iter_data(data=self.data)):
            ax = self._get_ax(i)
            if label is not None:
                label = pprint_thing(label)
                ax.set_ylabel(label)
            kwds = self.kwds.copy()

            def blank_labeler(label, value):
                if value == 0:
                    return ''
                else:
                    return label
            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop('labels', idx)
            if labels is not None:
                blabels = [blank_labeler(left, value) for left, value in zip(labels, y)]
            else:
                blabels = None
            results = ax.pie(y, labels=blabels, **kwds)
            if kwds.get('autopct', None) is not None:
                patches, texts, autotexts = results
            else:
                patches, texts = results
                autotexts = []
            if self.fontsize is not None:
                for t in texts + autotexts:
                    t.set_fontsize(self.fontsize)
            leglabels = labels if labels is not None else idx
            for _patch, _leglabel in zip(patches, leglabels):
                self._append_legend_handles_labels(_patch, _leglabel)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        pass