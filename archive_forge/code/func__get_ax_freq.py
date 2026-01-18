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
def _get_ax_freq(ax: Axes):
    """
    Get the freq attribute of the ax object if set.
    Also checks shared axes (eg when using secondary yaxis, sharex=True
    or twinx)
    """
    ax_freq = getattr(ax, 'freq', None)
    if ax_freq is None:
        if hasattr(ax, 'left_ax'):
            ax_freq = getattr(ax.left_ax, 'freq', None)
        elif hasattr(ax, 'right_ax'):
            ax_freq = getattr(ax.right_ax, 'freq', None)
    if ax_freq is None:
        shared_axes = ax.get_shared_x_axes().get_siblings(ax)
        if len(shared_axes) > 1:
            for shared_ax in shared_axes:
                ax_freq = getattr(shared_ax, 'freq', None)
                if ax_freq is not None:
                    break
    return ax_freq