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
class ImplementsArrayReduce:
    __slots__ = ()

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool, numeric_only: bool):
        if include_skipna:

            def wrapped_func(self, dim=None, axis=None, skipna=None, **kwargs):
                return self.reduce(func=func, dim=dim, axis=axis, skipna=skipna, **kwargs)
        else:

            def wrapped_func(self, dim=None, axis=None, **kwargs):
                return self.reduce(func=func, dim=dim, axis=axis, **kwargs)
        return wrapped_func
    _reduce_extra_args_docstring = dedent("        dim : str or sequence of str, optional\n            Dimension(s) over which to apply `{name}`.\n        axis : int or sequence of int, optional\n            Axis(es) over which to apply `{name}`. Only one of the 'dim'\n            and 'axis' arguments can be supplied. If neither are supplied, then\n            `{name}` is calculated over axes.")
    _cum_extra_args_docstring = dedent("        dim : str or sequence of str, optional\n            Dimension over which to apply `{name}`.\n        axis : int or sequence of int, optional\n            Axis over which to apply `{name}`. Only one of the 'dim'\n            and 'axis' arguments can be supplied.")