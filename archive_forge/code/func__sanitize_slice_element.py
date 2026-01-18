from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def _sanitize_slice_element(x):
    from xarray.core.dataarray import DataArray
    from xarray.core.variable import Variable
    if not isinstance(x, tuple) and len(np.shape(x)) != 0:
        raise ValueError(f'cannot use non-scalar arrays in a slice for xarray indexing: {x}')
    if isinstance(x, (Variable, DataArray)):
        x = x.values
    if isinstance(x, np.ndarray):
        x = x[()]
    return x