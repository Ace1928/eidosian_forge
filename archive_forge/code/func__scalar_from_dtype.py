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
def _scalar_from_dtype(dtype):
    if dtype.kind in ('i', 'f', 'u'):
        return dtype.type(1)
    elif dtype.kind == 'c':
        return dtype.type(complex(1, 0))
    elif dtype.kind in _simple_fake_mapping:
        o = _simple_fake_mapping[dtype.kind]
        return o.astype(dtype) if dtype.kind in ('m', 'M') else o
    else:
        raise TypeError(f"Can't handle dtype: {dtype}")