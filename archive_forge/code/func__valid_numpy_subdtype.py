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
def _valid_numpy_subdtype(x, numpy_types):
    """
    Is any dtype from numpy_types superior to the dtype of x?
    """
    for t in numpy_types:
        assert not np.issubdtype(np.generic, t)
    return any((np.issubdtype(x.dtype, t) for t in numpy_types))