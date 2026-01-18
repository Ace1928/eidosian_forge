from __future__ import annotations
import itertools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import make_na_array
class SingleArrayManager(BaseArrayManager, SingleDataManager):
    __slots__ = ['_axes', 'arrays']
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    @property
    def ndim(self) -> Literal[1]:
        return 1

    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool=True) -> None:
        self._axes = axes
        self.arrays = arrays
        if verify_integrity:
            assert len(axes) == 1
            assert len(arrays) == 1
            self._axes = [ensure_index(ax) for ax in self._axes]
            arr = arrays[0]
            arr = maybe_coerce_values(arr)
            arr = extract_pandas_array(arr, None, 1)[0]
            self.arrays = [arr]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        n_rows, = self.shape
        assert len(self.arrays) == 1
        arr = self.arrays[0]
        assert len(arr) == n_rows
        if not arr.ndim == 1:
            raise ValueError(f'Passed array should be 1-dimensional, got array with {arr.ndim} dimensions instead.')

    @staticmethod
    def _normalize_axis(axis):
        return axis

    def make_empty(self, axes=None) -> Self:
        """Return an empty ArrayManager with index/array of length 0"""
        if axes is None:
            axes = [Index([], dtype=object)]
        array: np.ndarray = np.array([], dtype=self.dtype)
        return type(self)([array], axes)

    @classmethod
    def from_array(cls, array, index) -> SingleArrayManager:
        return cls([array], [index])

    @property
    def axes(self) -> list[Index]:
        return self._axes

    @property
    def index(self) -> Index:
        return self._axes[0]

    @property
    def dtype(self):
        return self.array.dtype

    def external_values(self):
        """The array that Series.values returns"""
        return external_values(self.array)

    def internal_values(self):
        """The array that Series._values returns"""
        return self.array

    def array_values(self):
        """The array that Series.array returns"""
        arr = self.array
        if isinstance(arr, np.ndarray):
            arr = NumpyExtensionArray(arr)
        return arr

    @property
    def _can_hold_na(self) -> bool:
        if isinstance(self.array, np.ndarray):
            return self.array.dtype.kind not in 'iub'
        else:
            return self.array._can_hold_na

    @property
    def is_single_block(self) -> bool:
        return True

    def fast_xs(self, loc: int) -> SingleArrayManager:
        raise NotImplementedError('Use series._values[loc] instead')

    def get_slice(self, slobj: slice, axis: AxisInt=0) -> SingleArrayManager:
        if axis >= self.ndim:
            raise IndexError('Requested axis not found in manager')
        new_array = self.array[slobj]
        new_index = self.index._getitem_slice(slobj)
        return type(self)([new_array], [new_index], verify_integrity=False)

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> SingleArrayManager:
        new_array = self.array[indexer]
        new_index = self.index[indexer]
        return type(self)([new_array], [new_index])

    def apply(self, func, **kwargs) -> Self:
        if callable(func):
            new_array = func(self.array, **kwargs)
        else:
            new_array = getattr(self.array, func)(**kwargs)
        return type(self)([new_array], self._axes)

    def setitem(self, indexer, value, warn: bool=True) -> SingleArrayManager:
        """
        Set values with indexer.

        For SingleArrayManager, this backs s[indexer] = value

        See `setitem_inplace` for a version that works inplace and doesn't
        return a new Manager.
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f'Cannot set values with ndim > {self.ndim}')
        return self.apply_with_block('setitem', indexer=indexer, value=value)

    def idelete(self, indexer) -> SingleArrayManager:
        """
        Delete selected locations in-place (new array, same ArrayManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False
        self.arrays = [self.arrays[0][to_keep]]
        self._axes = [self._axes[0][to_keep]]
        return self

    def _get_data_subset(self, predicate: Callable) -> SingleArrayManager:
        if predicate(self.array):
            return type(self)(self.arrays, self._axes, verify_integrity=False)
        else:
            return self.make_empty()

    def set_values(self, values: ArrayLike) -> None:
        """
        Set (replace) the values of the SingleArrayManager in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current SingleArrayManager (length, dtype, etc).
        """
        self.arrays[0] = values

    def to_2d_mgr(self, columns: Index) -> ArrayManager:
        """
        Manager analogue of Series.to_frame
        """
        arrays = [self.arrays[0]]
        axes = [self.axes[0], columns]
        return ArrayManager(arrays, axes, verify_integrity=False)