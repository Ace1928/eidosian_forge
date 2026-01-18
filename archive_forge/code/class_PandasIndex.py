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
class PandasIndex(Index):
    """Wrap a pandas.Index as an xarray compatible index."""
    index: pd.Index
    dim: Hashable
    coord_dtype: Any
    __slots__ = ('index', 'dim', 'coord_dtype')

    def __init__(self, array: Any, dim: Hashable, coord_dtype: Any=None):
        index = safe_cast_to_index(array).copy()
        if index.name is None:
            index.name = dim
        self.index = index
        self.dim = dim
        if coord_dtype is None:
            coord_dtype = get_valid_numpy_dtype(index)
        self.coord_dtype = coord_dtype

    def _replace(self, index, dim=None, coord_dtype=None):
        if dim is None:
            dim = self.dim
        if coord_dtype is None:
            coord_dtype = self.coord_dtype
        return type(self)(index, dim, coord_dtype)

    @classmethod
    def from_variables(cls, variables: Mapping[Any, Variable], *, options: Mapping[str, Any]) -> PandasIndex:
        if len(variables) != 1:
            raise ValueError(f'PandasIndex only accepts one variable, found {len(variables)} variables')
        name, var = next(iter(variables.items()))
        if var.ndim == 0:
            raise ValueError(f'cannot set a PandasIndex from the scalar variable {name!r}, only 1-dimensional variables are supported. Note: you might want to use `obj.expand_dims({name!r})` to create a new dimension and turn {name!r} as an indexed dimension coordinate.')
        elif var.ndim != 1:
            raise ValueError(f'PandasIndex only accepts a 1-dimensional variable, variable {name!r} has {var.ndim} dimensions')
        dim = var.dims[0]
        data = getattr(var._data, 'array', var.data)
        if isinstance(var._data, PandasMultiIndexingAdapter):
            level = var._data.level
            if level is not None:
                data = var._data.array.get_level_values(level)
        obj = cls(data, dim, coord_dtype=var.dtype)
        assert not isinstance(obj.index, pd.MultiIndex)
        obj.index.name = name
        return obj

    @staticmethod
    def _concat_indexes(indexes, dim, positions=None) -> pd.Index:
        new_pd_index: pd.Index
        if not indexes:
            new_pd_index = pd.Index([])
        else:
            if not all((idx.dim == dim for idx in indexes)):
                dims = ','.join({f'{idx.dim!r}' for idx in indexes})
                raise ValueError(f'Cannot concatenate along dimension {dim!r} indexes with dimensions: {dims}')
            pd_indexes = [idx.index for idx in indexes]
            new_pd_index = pd_indexes[0].append(pd_indexes[1:])
            if positions is not None:
                indices = nputils.inverse_permutation(np.concatenate(positions))
                new_pd_index = new_pd_index.take(indices)
        return new_pd_index

    @classmethod
    def concat(cls, indexes: Sequence[Self], dim: Hashable, positions: Iterable[Iterable[int]] | None=None) -> Self:
        new_pd_index = cls._concat_indexes(indexes, dim, positions)
        if not indexes:
            coord_dtype = None
        else:
            coord_dtype = np.result_type(*[idx.coord_dtype for idx in indexes])
        return cls(new_pd_index, dim=dim, coord_dtype=coord_dtype)

    def create_variables(self, variables: Mapping[Any, Variable] | None=None) -> IndexVars:
        from xarray.core.variable import IndexVariable
        name = self.index.name
        attrs: Mapping[Hashable, Any] | None
        encoding: Mapping[Hashable, Any] | None
        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None
        data = PandasIndexingAdapter(self.index, dtype=self.coord_dtype)
        var = IndexVariable(self.dim, data, attrs=attrs, encoding=encoding)
        return {name: var}

    def to_pandas_index(self) -> pd.Index:
        return self.index

    def isel(self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]) -> PandasIndex | None:
        from xarray.core.variable import Variable
        indxr = indexers[self.dim]
        if isinstance(indxr, Variable):
            if indxr.dims != (self.dim,):
                return None
            else:
                indxr = indxr.data
        if not isinstance(indxr, slice) and is_scalar(indxr):
            return None
        return self._replace(self.index[indxr])

    def sel(self, labels: dict[Any, Any], method=None, tolerance=None) -> IndexSelResult:
        from xarray.core.dataarray import DataArray
        from xarray.core.variable import Variable
        if method is not None and (not isinstance(method, str)):
            raise TypeError('``method`` must be a string')
        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))
        if isinstance(label, slice):
            indexer = _query_slice(self.index, label, coord_name, method, tolerance)
        elif is_dict_like(label):
            raise ValueError('cannot use a dict-like object for selection on a dimension that does not have a MultiIndex')
        else:
            label_array = normalize_label(label, dtype=self.coord_dtype)
            if label_array.ndim == 0:
                label_value = as_scalar(label_array)
                if isinstance(self.index, pd.CategoricalIndex):
                    if method is not None:
                        raise ValueError("'method' is not supported when indexing using a CategoricalIndex.")
                    if tolerance is not None:
                        raise ValueError("'tolerance' is not supported when indexing using a CategoricalIndex.")
                    indexer = self.index.get_loc(label_value)
                elif method is not None:
                    indexer = get_indexer_nd(self.index, label_array, method, tolerance)
                    if np.any(indexer < 0):
                        raise KeyError(f'not all values found in index {coord_name!r}')
                else:
                    try:
                        indexer = self.index.get_loc(label_value)
                    except KeyError as e:
                        raise KeyError(f"not all values found in index {coord_name!r}. Try setting the `method` keyword argument (example: method='nearest').") from e
            elif label_array.dtype.kind == 'b':
                indexer = label_array
            else:
                indexer = get_indexer_nd(self.index, label_array, method, tolerance)
                if np.any(indexer < 0):
                    raise KeyError(f'not all values found in index {coord_name!r}')
            if isinstance(label, Variable):
                indexer = Variable(label.dims, indexer)
            elif isinstance(label, DataArray):
                indexer = DataArray(indexer, coords=label._coords, dims=label.dims)
        return IndexSelResult({self.dim: indexer})

    def equals(self, other: Index):
        if not isinstance(other, PandasIndex):
            return False
        return self.index.equals(other.index) and self.dim == other.dim

    def join(self, other: Self, how: str='inner') -> Self:
        if how == 'outer':
            index = self.index.union(other.index)
        else:
            index = self.index.intersection(other.index)
        coord_dtype = np.result_type(self.coord_dtype, other.coord_dtype)
        return type(self)(index, self.dim, coord_dtype=coord_dtype)

    def reindex_like(self, other: Self, method=None, tolerance=None) -> dict[Hashable, Any]:
        if not self.index.is_unique:
            raise ValueError(f'cannot reindex or align along dimension {self.dim!r} because the (pandas) index has duplicate values')
        return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

    def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
        shift = shifts[self.dim] % self.index.shape[0]
        if shift != 0:
            new_pd_idx = self.index[-shift:].append(self.index[:-shift])
        else:
            new_pd_idx = self.index[:]
        return self._replace(new_pd_idx)

    def rename(self, name_dict, dims_dict):
        if self.index.name not in name_dict and self.dim not in dims_dict:
            return self
        new_name = name_dict.get(self.index.name, self.index.name)
        index = self.index.rename(new_name)
        new_dim = dims_dict.get(self.dim, self.dim)
        return self._replace(index, dim=new_dim)

    def _copy(self: T_PandasIndex, deep: bool=True, memo: dict[int, Any] | None=None) -> T_PandasIndex:
        if deep:
            index = self.index.copy(deep=True)
        else:
            index = self.index
        return self._replace(index)

    def __getitem__(self, indexer: Any):
        return self._replace(self.index[indexer])

    def __repr__(self):
        return f'PandasIndex({repr(self.index)})'