from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _set_concise_date(ax: Axes, axis: Literal['x', 'y', 'z']='x') -> None:
    """
    Use ConciseDateFormatter which is meant to improve the
    strings chosen for the ticklabels, and to minimize the
    strings used in those tick labels as much as possible.

    https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html

    Parameters
    ----------
    ax : Axes
        Figure axes.
    axis : Literal["x", "y", "z"], optional
        Which axis to make concise. The default is "x".
    """
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    _axis = getattr(ax, f'{axis}axis')
    _axis.set_major_locator(locator)
    _axis.set_major_formatter(formatter)