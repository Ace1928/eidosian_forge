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
class TimeAccessor(Generic[T_DataArray]):
    __slots__ = ('_obj',)

    def __init__(self, obj: T_DataArray) -> None:
        self._obj = obj

    def _date_field(self, name: str, dtype: DTypeLike) -> T_DataArray:
        if dtype is None:
            dtype = self._obj.dtype
        result = _get_date_field(_index_or_data(self._obj), name, dtype)
        newvar = self._obj.variable.copy(data=result, deep=False)
        return self._obj._replace(newvar, name=name)

    def _tslib_round_accessor(self, name: str, freq: str) -> T_DataArray:
        result = _round_field(_index_or_data(self._obj), name, freq)
        newvar = self._obj.variable.copy(data=result, deep=False)
        return self._obj._replace(newvar, name=name)

    def floor(self, freq: str) -> T_DataArray:
        """
        Round timestamps downward to specified frequency resolution.

        Parameters
        ----------
        freq : str
            a freq string indicating the rounding resolution e.g. "D" for daily resolution

        Returns
        -------
        floor-ed timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        """
        return self._tslib_round_accessor('floor', freq)

    def ceil(self, freq: str) -> T_DataArray:
        """
        Round timestamps upward to specified frequency resolution.

        Parameters
        ----------
        freq : str
            a freq string indicating the rounding resolution e.g. "D" for daily resolution

        Returns
        -------
        ceil-ed timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        """
        return self._tslib_round_accessor('ceil', freq)

    def round(self, freq: str) -> T_DataArray:
        """
        Round timestamps to specified frequency resolution.

        Parameters
        ----------
        freq : str
            a freq string indicating the rounding resolution e.g. "D" for daily resolution

        Returns
        -------
        rounded timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        """
        return self._tslib_round_accessor('round', freq)