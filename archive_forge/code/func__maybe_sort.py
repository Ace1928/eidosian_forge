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
def _maybe_sort(a, check_index: bool):
    try:
        if is_dataframe_like(a):
            if set(a.index.names) & set(a.columns):
                a.index.names = ['-overlapped-index-name-%d' % i for i in range(len(a.index.names))]
            a = a.sort_values(by=methods.tolist(a.columns))
        else:
            a = a.sort_values()
    except (TypeError, IndexError, ValueError):
        pass
    return a.sort_index() if check_index else a