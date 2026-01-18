import functools
import warnings
from collections.abc import Hashable
from typing import Union
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable
def _assert_variable_invariants(var: Variable, name: Hashable=None):
    if name is None:
        name_or_empty: tuple = ()
    else:
        name_or_empty = (name,)
    assert isinstance(var._dims, tuple), name_or_empty + (var._dims,)
    assert len(var._dims) == len(var._data.shape), name_or_empty + (var._dims, var._data.shape)
    assert isinstance(var._encoding, (type(None), dict)), name_or_empty + (var._encoding,)
    assert isinstance(var._attrs, (type(None), dict)), name_or_empty + (var._attrs,)