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
def _update_axes(ax: Axes, xincrease: bool | None, yincrease: bool | None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None) -> None:
    """
    Update axes with provided parameters
    """
    if xincrease is None:
        pass
    elif xincrease and ax.xaxis_inverted():
        ax.invert_xaxis()
    elif not xincrease and (not ax.xaxis_inverted()):
        ax.invert_xaxis()
    if yincrease is None:
        pass
    elif yincrease and ax.yaxis_inverted():
        ax.invert_yaxis()
    elif not yincrease and (not ax.yaxis_inverted()):
        ax.invert_yaxis()
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)