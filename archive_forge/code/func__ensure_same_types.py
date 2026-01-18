from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
def _ensure_same_types(series, dim):
    if series.dtype == object:
        types = set(series.map(type))
        if len(types) > 1:
            try:
                import cftime
                cftimes = any((issubclass(t, cftime.datetime) for t in types))
            except ImportError:
                cftimes = False
            types = ', '.join((t.__name__ for t in types))
            error_msg = f"Cannot combine along dimension '{dim}' with mixed types. Found: {types}."
            if cftimes:
                error_msg = f'{error_msg} If importing data directly from a file then setting `use_cftime=True` may fix this issue.'
            raise TypeError(error_msg)