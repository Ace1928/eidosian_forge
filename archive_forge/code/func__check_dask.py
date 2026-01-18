from __future__ import annotations
import math
import re
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Number
from typing import TypeVar, overload
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
import dask
from dask.base import get_scheduler, is_dask_collection
from dask.core import get_deps
from dask.dataframe import (  # noqa: F401 register pandas extension types
from dask.dataframe._compat import PANDAS_GE_150, tm  # noqa: F401
from dask.dataframe.dispatch import (  # noqa : F401
from dask.dataframe.extensions import make_scalar
from dask.typing import NoDefault, no_default
from dask.utils import (
def _check_dask(dsk, check_names=True, check_dtypes=True, result=None, scheduler=None):
    import dask.dataframe as dd
    if hasattr(dsk, '__dask_graph__'):
        graph = dsk.__dask_graph__()
        if hasattr(graph, 'validate'):
            graph.validate()
        if result is None:
            result = dsk.compute(scheduler=scheduler)
        if isinstance(dsk, dd.Index) or is_index_like(dsk._meta):
            assert 'Index' in type(result).__name__, type(result)
            if check_names:
                assert dsk.name == result.name
                assert dsk._meta.name == result.name
                if isinstance(result, pd.MultiIndex):
                    assert result.names == dsk._meta.names
            if check_dtypes:
                assert_dask_dtypes(dsk, result)
        elif isinstance(dsk, dd.Series) or is_series_like(dsk._meta):
            assert 'Series' in type(result).__name__, type(result)
            assert type(dsk._meta) == type(result), type(dsk._meta)
            if check_names:
                assert dsk.name == result.name, (dsk.name, result.name)
                assert dsk._meta.name == result.name
            if check_dtypes:
                assert_dask_dtypes(dsk, result)
            _check_dask(dsk.index, check_names=check_names, check_dtypes=check_dtypes, result=result.index)
        elif isinstance(dsk, dd.DataFrame) or is_dataframe_like(dsk._meta):
            assert 'DataFrame' in type(result).__name__, type(result)
            assert isinstance(dsk.columns, pd.Index), type(dsk.columns)
            assert type(dsk._meta) == type(result), type(dsk._meta)
            if check_names:
                tm.assert_index_equal(dsk.columns, result.columns)
                tm.assert_index_equal(dsk._meta.columns, result.columns)
            if check_dtypes:
                assert_dask_dtypes(dsk, result)
            _check_dask(dsk.index, check_names=check_names, check_dtypes=check_dtypes, result=result.index)
        else:
            if not np.isscalar(result) and (not isinstance(result, (pd.Timestamp, pd.Timedelta))):
                raise TypeError('Expected object of type dataframe, series, index, or scalar.\n    Got: ' + str(type(result)))
            if check_dtypes:
                assert_dask_dtypes(dsk, result)
        return result
    return dsk