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
def _upsample_others(ax: Axes, freq: BaseOffset, kwargs: dict[str, Any]) -> None:
    legend = ax.get_legend()
    lines, labels = _replot_ax(ax, freq)
    _replot_ax(ax, freq)
    other_ax = None
    if hasattr(ax, 'left_ax'):
        other_ax = ax.left_ax
    if hasattr(ax, 'right_ax'):
        other_ax = ax.right_ax
    if other_ax is not None:
        rlines, rlabels = _replot_ax(other_ax, freq)
        lines.extend(rlines)
        labels.extend(rlabels)
    if legend is not None and kwargs.get('legend', True) and (len(lines) > 0):
        title: str | None = legend.get_title().get_text()
        if title == 'None':
            title = None
        ax.legend(lines, labels, loc='best', title=title)