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
def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors_i = np.linspace(0, 1.0, n_colors)
    if isinstance(cmap, (list, tuple)):
        cmap = ListedColormap(cmap, N=n_colors)
        pal = cmap(colors_i)
    elif isinstance(cmap, str):
        try:
            cmap = plt.get_cmap(cmap)
            pal = cmap(colors_i)
        except ValueError:
            try:
                from seaborn import color_palette
                pal = color_palette(cmap, n_colors=n_colors)
            except (ValueError, ImportError):
                cmap = ListedColormap([cmap], N=n_colors)
                pal = cmap(colors_i)
    else:
        pal = cmap(colors_i)
    return pal