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
def _assert_valid_xy(darray: DataArray | Dataset, xy: Hashable | None, name: str) -> None:
    """
    make sure x and y passed to plotting functions are valid
    """
    multiindex_dims = {idx.dim for idx in darray.xindexes.get_unique() if isinstance(idx, PandasMultiIndex)}
    valid_xy = (set(darray.dims) | set(darray.coords)) - multiindex_dims
    if xy is not None and xy not in valid_xy:
        valid_xy_str = "', '".join(sorted(tuple((str(v) for v in valid_xy))))
        raise ValueError(f"{name} must be one of None, '{valid_xy_str}'. Received '{xy}' instead.")