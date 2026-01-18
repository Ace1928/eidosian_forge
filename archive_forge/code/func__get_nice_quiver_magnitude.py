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
def _get_nice_quiver_magnitude(u, v):
    import matplotlib as mpl
    ticker = mpl.ticker.MaxNLocator(3)
    mean = np.mean(np.hypot(u.to_numpy(), v.to_numpy()))
    magnitude = ticker.tick_values(0, mean)[-2]
    return magnitude