from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Generic
import numpy as np
import pandas as pd
from xarray.coding.times import infer_calendar_name
from xarray.core import duck_array_ops
from xarray.core.common import (
from xarray.core.types import T_DataArray
from xarray.core.variable import IndexVariable
from xarray.namedarray.utils import is_duck_dask_array
def _tslib_round_accessor(self, name: str, freq: str) -> T_DataArray:
    result = _round_field(_index_or_data(self._obj), name, freq)
    newvar = self._obj.variable.copy(data=result, deep=False)
    return self._obj._replace(newvar, name=name)