from __future__ import annotations
import numbers
import string
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype
import pandas as pd
from pandas.api.types import (
from pandas.core.arrays import ExtensionArray
class ListArray(ExtensionArray):
    dtype = ListDtype()
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False) -> None:
        if not isinstance(values, np.ndarray):
            raise TypeError('Need to pass a numpy array as values')
        for val in values:
            if not isinstance(val, self.dtype.type) and (not pd.isna(val)):
                raise TypeError('All values must be of type ' + str(self.dtype.type))
        self.data = values

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        data = np.empty(len(scalars), dtype=object)
        data[:] = scalars
        return cls(data)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            return type(self)(self.data[item])

    def __len__(self) -> int:
        return len(self.data)

    def isna(self):
        return np.array([not isinstance(x, list) and np.isnan(x) for x in self.data], dtype=bool)

    def take(self, indexer, allow_fill=False, fill_value=None):
        indexer = np.asarray(indexer)
        msg = 'Index is out of bounds or cannot do a non-empty take from an empty array.'
        if allow_fill:
            if fill_value is None:
                fill_value = self.dtype.na_value
            if (indexer < -1).any():
                raise ValueError
            try:
                output = [self.data[loc] if loc != -1 else fill_value for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err
        else:
            try:
                output = [self.data[loc] for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err
        return self._from_sequence(output)

    def copy(self):
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif is_string_dtype(dtype) and (not is_object_dtype(dtype)):
            return np.array([str(x) for x in self.data], dtype=dtype)
        return np.array(self.data, dtype=dtype, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([x.data for x in to_concat])
        return cls(data)