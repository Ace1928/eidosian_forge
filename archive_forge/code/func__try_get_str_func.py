from __future__ import annotations
import warnings
from collections.abc import Iterable
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
import pandas.core.common as com
import pandas.core.groupby
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.dtypes.common import (
from pandas.errors import SpecificationError
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import (
from .series import Series
from .utils import is_label
from .window import RollingGroupby
def _try_get_str_func(self, fn):
    """
        Try to convert a groupby aggregation function to a string or list of such.

        Parameters
        ----------
        fn : callable, str, or Iterable

        Returns
        -------
        str, list
            If `fn` is a callable, return its name, otherwise return `fn` itself.
            If `fn` is a string, return it. If `fn` is an Iterable, return a list
            of _try_get_str_func applied to each element of `fn`.
        """
    if not isinstance(fn, str) and isinstance(fn, Iterable):
        return [self._try_get_str_func(f) for f in fn]
    return fn.__name__ if callable(fn) else fn