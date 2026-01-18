from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
class IndexVariable(Variable):
    """Wrapper for accommodating a pandas.Index in an xarray.Variable.

    IndexVariable preserve loaded values in the form of a pandas.Index instead
    of a NumPy array. Hence, their values are immutable and must always be one-
    dimensional.

    They also have a name property, which is the name of their sole dimension
    unless another name is given.
    """
    __slots__ = ()
    _data: PandasIndexingAdapter

    def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
        super().__init__(dims, data, attrs, encoding, fastpath)
        if self.ndim != 1:
            raise ValueError(f'{type(self).__name__} objects must be 1-dimensional')
        if not isinstance(self._data, PandasIndexingAdapter):
            self._data = PandasIndexingAdapter(self._data)

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token
        return normalize_token((type(self), self._dims, self._data.array, self._attrs or None))

    def load(self):
        return self

    @Variable.data.setter
    def data(self, data):
        raise ValueError(f'Cannot assign to the .data attribute of dimension coordinate a.k.a IndexVariable {self.name!r}. Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate.')

    @Variable.values.setter
    def values(self, values):
        raise ValueError(f'Cannot assign to the .values attribute of dimension coordinate a.k.a IndexVariable {self.name!r}. Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate.')

    def chunk(self, chunks={}, name=None, lock=False, inline_array=False, chunked_array_type=None, from_array_kwargs=None):
        return self.copy(deep=False)

    def _as_sparse(self, sparse_format=_default, fill_value=_default):
        return self.copy(deep=False)

    def _to_dense(self):
        return self.copy(deep=False)

    def _finalize_indexing_result(self, dims, data):
        if getattr(data, 'ndim', 0) != 1:
            return Variable(dims, data, self._attrs, self._encoding)
        else:
            return self._replace(dims=dims, data=data)

    def __setitem__(self, key, value):
        raise TypeError(f'{type(self).__name__} values cannot be modified')

    @classmethod
    def concat(cls, variables, dim='concat_dim', positions=None, shortcut=False, combine_attrs='override'):
        """Specialized version of Variable.concat for IndexVariable objects.

        This exists because we want to avoid converting Index objects to NumPy
        arrays, if possible.
        """
        from xarray.core.merge import merge_attrs
        if not isinstance(dim, str):
            dim, = dim.dims
        variables = list(variables)
        first_var = variables[0]
        if any((not isinstance(v, cls) for v in variables)):
            raise TypeError('IndexVariable.concat requires that all input variables be IndexVariable objects')
        indexes = [v._data.array for v in variables]
        if not indexes:
            data = []
        else:
            data = indexes[0].append(indexes[1:])
            if positions is not None:
                indices = nputils.inverse_permutation(np.concatenate(positions))
                data = data.take(indices)
        data = maybe_coerce_to_str(data, variables)
        attrs = merge_attrs([var.attrs for var in variables], combine_attrs=combine_attrs)
        if not shortcut:
            for var in variables:
                if var.dims != first_var.dims:
                    raise ValueError('inconsistent dimensions')
        return cls(first_var.dims, data, attrs)

    def copy(self, deep: bool=True, data: T_DuckArray | ArrayLike | None=None):
        """Returns a copy of this object.

        `deep` is ignored since data is stored in the form of
        pandas.Index, which is already immutable. Dimensions, attributes
        and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: True
            Deep is ignored when data is given. Whether the data array is
            loaded into memory and copied onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.

        Returns
        -------
        object : Variable
            New object with dimensions, attributes, encodings, and optionally
            data copied from original.
        """
        if data is None:
            ndata = self._data
            if deep:
                ndata = copy.deepcopy(ndata, None)
        else:
            ndata = as_compatible_data(data)
            if self.shape != ndata.shape:
                raise ValueError(f'Data shape {ndata.shape} must match shape of object {self.shape}')
        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)
        encoding = copy.deepcopy(self._encoding) if deep else copy.copy(self._encoding)
        return self._replace(data=ndata, attrs=attrs, encoding=encoding)

    def equals(self, other, equiv=None):
        if equiv is not None:
            return super().equals(other, equiv)
        other = getattr(other, 'variable', other)
        try:
            return self.dims == other.dims and self._data_equals(other)
        except (TypeError, AttributeError):
            return False

    def _data_equals(self, other):
        return self._to_index().equals(other._to_index())

    def to_index_variable(self) -> IndexVariable:
        """Return this variable as an xarray.IndexVariable"""
        return self.copy(deep=False)
    to_coord = utils.alias(to_index_variable, 'to_coord')

    def _to_index(self) -> pd.Index:
        assert self.ndim == 1
        index = self._data.array
        if isinstance(index, pd.MultiIndex):
            valid_level_names = [name or f'{self.dims[0]}_level_{i}' for i, name in enumerate(index.names)]
            index = index.set_names(valid_level_names)
        else:
            index = index.set_names(self.name)
        return index

    def to_index(self) -> pd.Index:
        """Convert this variable to a pandas.Index"""
        index = self._to_index()
        level = getattr(self._data, 'level', None)
        if level is not None:
            return index.get_level_values(level)
        else:
            return index

    @property
    def level_names(self) -> list[str] | None:
        """Return MultiIndex level names or None if this IndexVariable has no
        MultiIndex.
        """
        index = self.to_index()
        if isinstance(index, pd.MultiIndex):
            return index.names
        else:
            return None

    def get_level_variable(self, level):
        """Return a new IndexVariable from a given MultiIndex level."""
        if self.level_names is None:
            raise ValueError(f'IndexVariable {self.name!r} has no MultiIndex')
        index = self.to_index()
        return type(self)(self.dims, index.get_level_values(level))

    @property
    def name(self) -> Hashable:
        return self.dims[0]

    @name.setter
    def name(self, value) -> NoReturn:
        raise AttributeError('cannot modify name of IndexVariable in-place')

    def _inplace_binary_op(self, other, f):
        raise TypeError('Values of an IndexVariable are immutable and can not be modified inplace')