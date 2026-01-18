from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _flatten_visible(axes: Axes | Sequence[Axes]) -> Sequence[Axes]:
    """
    Flatten axes, and filter only visible

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like

    """
    from pandas.plotting._matplotlib.tools import flatten_axes
    axes_ndarray = flatten_axes(axes)
    axes = [ax for ax in axes_ndarray if ax.get_visible()]
    return axes