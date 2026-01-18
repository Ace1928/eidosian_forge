from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def _contains_cftime_datetimes(array: Any) -> bool:
    """Check if a array inside a Variable contains cftime.datetime objects"""
    if cftime is None:
        return False
    if array.dtype == np.dtype('O') and array.size > 0:
        first_idx = (0,) * array.ndim
        if isinstance(array, ExplicitlyIndexed):
            first_idx = BasicIndexer(first_idx)
        sample = array[first_idx]
        return isinstance(np.asarray(sample).item(), cftime.datetime)
    return False