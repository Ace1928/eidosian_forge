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
def _add_legend(hueplt_norm: _Normalize, sizeplt_norm: _Normalize, primitive, legend_ax, plotfunc: str):
    primitive = primitive if isinstance(primitive, list) else [primitive]
    handles, labels = ([], [])
    for huesizeplt, prop in [(hueplt_norm, 'colors'), (sizeplt_norm, 'sizes')]:
        if huesizeplt.data is not None:
            hdl, lbl = ([], [])
            for p in primitive:
                hdl_, lbl_ = legend_elements(p, prop, num='auto', func=huesizeplt.func)
                hdl += hdl_
                lbl += lbl_
            u, ind = np.unique(lbl, return_index=True)
            ind = np.argsort(ind)
            lbl = u[ind].tolist()
            hdl = np.array(hdl)[ind].tolist()
            hdl, lbl = _legend_add_subtitle(hdl, lbl, label_from_attrs(huesizeplt.data))
            handles += hdl
            labels += lbl
    legend = legend_ax.legend(handles, labels, framealpha=0.5)
    _adjust_legend_subtitles(legend)
    return legend