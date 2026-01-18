from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
class ExtensionBlock(EABackedBlock):
    """
    Block for holding extension types.

    Notes
    -----
    This holds all 3rd-party extension array types. It's also the immediate
    parent class for our internal extension types' blocks.

    ExtensionArrays are limited to 1-D.
    """
    values: ExtensionArray

    def fillna(self, value, limit: int | None=None, inplace: bool=False, downcast=None, using_cow: bool=False, already_warned=None) -> list[Block]:
        if isinstance(self.dtype, IntervalDtype):
            return super().fillna(value=value, limit=limit, inplace=inplace, downcast=downcast, using_cow=using_cow, already_warned=already_warned)
        if using_cow and self._can_hold_na and (not self.values._hasna):
            refs = self.refs
            new_values = self.values
        else:
            copy, refs = self._get_refs_and_copy(using_cow, inplace)
            try:
                new_values = self.values.fillna(value=value, method=None, limit=limit, copy=copy)
            except TypeError:
                refs = None
                new_values = self.values.fillna(value=value, method=None, limit=limit)
                warnings.warn(f"ExtensionArray.fillna added a 'copy' keyword in pandas 2.1.0. In a future version, ExtensionArray subclasses will need to implement this keyword or an exception will be raised. In the interim, the keyword is ignored by {type(self.values).__name__}.", DeprecationWarning, stacklevel=find_stack_level())
            else:
                if not copy and warn_copy_on_write() and (already_warned is not None) and (not already_warned.warned_already):
                    if self.refs.has_reference():
                        warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())
                        already_warned.warned_already = True
        nb = self.make_block_same_class(new_values, refs=refs)
        return nb._maybe_downcast([nb], downcast, using_cow=using_cow, caller='fillna')

    @cache_readonly
    def shape(self) -> Shape:
        if self.ndim == 1:
            return (len(self.values),)
        return (len(self._mgr_locs), len(self.values))

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]):
        if isinstance(i, tuple):
            col, loc = i
            if not com.is_null_slice(col) and col != 0:
                raise IndexError(f'{self} only contains one item')
            if isinstance(col, slice):
                if loc < 0:
                    loc += len(self.values)
                return self.values[loc:loc + 1]
            return self.values[loc]
        else:
            if i != 0:
                raise IndexError(f'{self} only contains one item')
            return self.values

    def set_inplace(self, locs, values: ArrayLike, copy: bool=False) -> None:
        if copy:
            self.values = self.values.copy()
        self.values[:] = values

    def _maybe_squeeze_arg(self, arg):
        """
        If necessary, squeeze a (N, 1) ndarray to (N,)
        """
        if isinstance(arg, (np.ndarray, ExtensionArray)) and arg.ndim == self.values.ndim + 1:
            assert arg.shape[1] == 1
            arg = arg[:, 0]
        elif isinstance(arg, ABCDataFrame):
            assert arg.shape[1] == 1
            arg = arg._ixs(0, axis=1)._values
        return arg

    def _unwrap_setitem_indexer(self, indexer):
        """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
        if isinstance(indexer, tuple) and len(indexer) == 2:
            if all((isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer)):
                first, second = indexer
                if not (second.size == 1 and (second == 0).all() and (first.shape[1] == 1)):
                    raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/')
                indexer = first[:, 0]
            elif lib.is_integer(indexer[1]) and indexer[1] == 0:
                indexer = indexer[0]
            elif com.is_null_slice(indexer[1]):
                indexer = indexer[0]
            elif is_list_like(indexer[1]) and indexer[1][0] == 0:
                indexer = indexer[0]
            else:
                raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/')
        return indexer

    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
        return False

    @cache_readonly
    def is_numeric(self) -> bool:
        return self.values.dtype._is_numeric

    def _slice(self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]) -> ExtensionArray:
        """
        Return a slice of my values.

        Parameters
        ----------
        slicer : slice, ndarray[int], or ndarray[bool]
            Valid (non-reducing) indexer for self.values.

        Returns
        -------
        ExtensionArray
        """
        if self.ndim == 2:
            if not isinstance(slicer, slice):
                raise AssertionError('invalid slicing for a 1-ndim ExtensionArray', slicer)
            new_locs = range(1)[slicer]
            if not len(new_locs):
                raise AssertionError('invalid slicing for a 1-ndim ExtensionArray', slicer)
            slicer = slice(None)
        return self.values[slicer]

    @final
    def slice_block_rows(self, slicer: slice) -> Self:
        """
        Perform __getitem__-like specialized to slicing along index.
        """
        new_values = self.values[slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)

    def _unstack(self, unstacker, fill_value, new_placement: npt.NDArray[np.intp], needs_masking: npt.NDArray[np.bool_]):
        new_values, mask = unstacker.arange_result
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]
        blocks = [type(self)(self.values.take(indices, allow_fill=needs_masking[i], fill_value=fill_value), BlockPlacement(place), ndim=2) for i, (indices, place) in enumerate(zip(new_values, new_placement))]
        return (blocks, mask)