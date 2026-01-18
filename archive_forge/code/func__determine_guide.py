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
def _determine_guide(hueplt_norm: _Normalize, sizeplt_norm: _Normalize, add_colorbar: None | bool=None, add_legend: None | bool=None, plotfunc_name: str | None=None) -> tuple[bool, bool]:
    if plotfunc_name == 'hist':
        return (False, False)
    if add_colorbar and hueplt_norm.data is None:
        raise KeyError('Cannot create a colorbar when hue is None.')
    if add_colorbar is None:
        if hueplt_norm.data is not None:
            add_colorbar = True
        else:
            add_colorbar = False
    if add_legend and hueplt_norm.data is None and (sizeplt_norm.data is None):
        raise KeyError('Cannot create a legend when hue and markersize is None.')
    if add_legend is None:
        if not add_colorbar and (hueplt_norm.data is not None and hueplt_norm.data_is_numeric is False) or sizeplt_norm.data is not None:
            add_legend = True
        else:
            add_legend = False
    return (add_colorbar, add_legend)