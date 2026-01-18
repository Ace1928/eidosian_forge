from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
class _Concatenator:
    """
    Orchestrates a concatenation operation for BlockManagers
    """
    sort: bool

    def __init__(self, objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame], axis: Axis=0, join: str='outer', keys: Iterable[Hashable] | None=None, levels=None, names: list[HashableT] | None=None, ignore_index: bool=False, verify_integrity: bool=False, copy: bool=True, sort: bool=False) -> None:
        if isinstance(objs, (ABCSeries, ABCDataFrame, str)):
            raise TypeError(f'first argument must be an iterable of pandas objects, you passed an object of type "{type(objs).__name__}"')
        if join == 'outer':
            self.intersect = False
        elif join == 'inner':
            self.intersect = True
        else:
            raise ValueError('Only can inner (intersect) or outer (union) join the other axis')
        if not is_bool(sort):
            raise ValueError(f"The 'sort' keyword only accepts boolean values; {sort} was passed.")
        self.sort = sort
        self.ignore_index = ignore_index
        self.verify_integrity = verify_integrity
        self.copy = copy
        objs, keys = self._clean_keys_and_objs(objs, keys)
        ndims = self._get_ndims(objs)
        sample, objs = self._get_sample_object(objs, ndims, keys, names, levels)
        if sample.ndim == 1:
            from pandas import DataFrame
            axis = DataFrame._get_axis_number(axis)
            self._is_frame = False
            self._is_series = True
        else:
            axis = sample._get_axis_number(axis)
            self._is_frame = True
            self._is_series = False
            axis = sample._get_block_manager_axis(axis)
        if len(ndims) > 1:
            objs = self._sanitize_mixed_ndim(objs, sample, ignore_index, axis)
        self.objs = objs
        self.bm_axis = axis
        self.axis = 1 - self.bm_axis if self._is_frame else 0
        self.keys = keys
        self.names = names or getattr(keys, 'names', None)
        self.levels = levels

    def _get_ndims(self, objs: list[Series | DataFrame]) -> set[int]:
        ndims = set()
        for obj in objs:
            if not isinstance(obj, (ABCSeries, ABCDataFrame)):
                msg = f"cannot concatenate object of type '{type(obj)}'; only Series and DataFrame objs are valid"
                raise TypeError(msg)
            ndims.add(obj.ndim)
        return ndims

    def _clean_keys_and_objs(self, objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame], keys) -> tuple[list[Series | DataFrame], Index | None]:
        if isinstance(objs, abc.Mapping):
            if keys is None:
                keys = list(objs.keys())
            objs_list = [objs[k] for k in keys]
        else:
            objs_list = list(objs)
        if len(objs_list) == 0:
            raise ValueError('No objects to concatenate')
        if keys is None:
            objs_list = list(com.not_none(*objs_list))
        else:
            clean_keys = []
            clean_objs = []
            if is_iterator(keys):
                keys = list(keys)
            if len(keys) != len(objs_list):
                warnings.warn('The behavior of pd.concat with len(keys) != len(objs) is deprecated. In a future version this will raise instead of truncating to the smaller of the two sequences', FutureWarning, stacklevel=find_stack_level())
            for k, v in zip(keys, objs_list):
                if v is None:
                    continue
                clean_keys.append(k)
                clean_objs.append(v)
            objs_list = clean_objs
            if isinstance(keys, MultiIndex):
                keys = type(keys).from_tuples(clean_keys, names=keys.names)
            else:
                name = getattr(keys, 'name', None)
                keys = Index(clean_keys, name=name, dtype=getattr(keys, 'dtype', None))
        if len(objs_list) == 0:
            raise ValueError('All objects passed were None')
        return (objs_list, keys)

    def _get_sample_object(self, objs: list[Series | DataFrame], ndims: set[int], keys, names, levels) -> tuple[Series | DataFrame, list[Series | DataFrame]]:
        sample: Series | DataFrame | None = None
        if len(ndims) > 1:
            max_ndim = max(ndims)
            for obj in objs:
                if obj.ndim == max_ndim and np.sum(obj.shape):
                    sample = obj
                    break
        else:
            non_empties = [obj for obj in objs if sum(obj.shape) > 0 or obj.ndim == 1]
            if len(non_empties) and (keys is None and names is None and (levels is None) and (not self.intersect)):
                objs = non_empties
                sample = objs[0]
        if sample is None:
            sample = objs[0]
        return (sample, objs)

    def _sanitize_mixed_ndim(self, objs: list[Series | DataFrame], sample: Series | DataFrame, ignore_index: bool, axis: AxisInt) -> list[Series | DataFrame]:
        new_objs = []
        current_column = 0
        max_ndim = sample.ndim
        for obj in objs:
            ndim = obj.ndim
            if ndim == max_ndim:
                pass
            elif ndim != max_ndim - 1:
                raise ValueError('cannot concatenate unaligned mixed dimensional NDFrame objects')
            else:
                name = getattr(obj, 'name', None)
                if ignore_index or name is None:
                    if axis == 1:
                        name = 0
                    else:
                        name = current_column
                        current_column += 1
                obj = sample._constructor({name: obj}, copy=False)
            new_objs.append(obj)
        return new_objs

    def get_result(self):
        cons: Callable[..., DataFrame | Series]
        sample: DataFrame | Series
        if self._is_series:
            sample = cast('Series', self.objs[0])
            if self.bm_axis == 0:
                name = com.consensus_name_attr(self.objs)
                cons = sample._constructor
                arrs = [ser._values for ser in self.objs]
                res = concat_compat(arrs, axis=0)
                new_index: Index
                if self.ignore_index:
                    new_index = default_index(len(res))
                else:
                    new_index = self.new_axes[0]
                mgr = type(sample._mgr).from_array(res, index=new_index)
                result = sample._constructor_from_mgr(mgr, axes=mgr.axes)
                result._name = name
                return result.__finalize__(self, method='concat')
            else:
                data = dict(zip(range(len(self.objs)), self.objs))
                cons = sample._constructor_expanddim
                index, columns = self.new_axes
                df = cons(data, index=index, copy=self.copy)
                df.columns = columns
                return df.__finalize__(self, method='concat')
        else:
            sample = cast('DataFrame', self.objs[0])
            mgrs_indexers = []
            for obj in self.objs:
                indexers = {}
                for ax, new_labels in enumerate(self.new_axes):
                    if ax == self.bm_axis:
                        continue
                    obj_labels = obj.axes[1 - ax]
                    if not new_labels.equals(obj_labels):
                        indexers[ax] = obj_labels.get_indexer(new_labels)
                mgrs_indexers.append((obj._mgr, indexers))
            new_data = concatenate_managers(mgrs_indexers, self.new_axes, concat_axis=self.bm_axis, copy=self.copy)
            if not self.copy and (not using_copy_on_write()):
                new_data._consolidate_inplace()
            out = sample._constructor_from_mgr(new_data, axes=new_data.axes)
            return out.__finalize__(self, method='concat')

    def _get_result_dim(self) -> int:
        if self._is_series and self.bm_axis == 1:
            return 2
        else:
            return self.objs[0].ndim

    @cache_readonly
    def new_axes(self) -> list[Index]:
        ndim = self._get_result_dim()
        return [self._get_concat_axis if i == self.bm_axis else self._get_comb_axis(i) for i in range(ndim)]

    def _get_comb_axis(self, i: AxisInt) -> Index:
        data_axis = self.objs[0]._get_block_manager_axis(i)
        return get_objs_combined_axis(self.objs, axis=data_axis, intersect=self.intersect, sort=self.sort, copy=self.copy)

    @cache_readonly
    def _get_concat_axis(self) -> Index:
        """
        Return index to be used along concatenation axis.
        """
        if self._is_series:
            if self.bm_axis == 0:
                indexes = [x.index for x in self.objs]
            elif self.ignore_index:
                idx = default_index(len(self.objs))
                return idx
            elif self.keys is None:
                names: list[Hashable] = [None] * len(self.objs)
                num = 0
                has_names = False
                for i, x in enumerate(self.objs):
                    if x.ndim != 1:
                        raise TypeError(f"Cannot concatenate type 'Series' with object of type '{type(x).__name__}'")
                    if x.name is not None:
                        names[i] = x.name
                        has_names = True
                    else:
                        names[i] = num
                        num += 1
                if has_names:
                    return Index(names)
                else:
                    return default_index(len(self.objs))
            else:
                return ensure_index(self.keys).set_names(self.names)
        else:
            indexes = [x.axes[self.axis] for x in self.objs]
        if self.ignore_index:
            idx = default_index(sum((len(i) for i in indexes)))
            return idx
        if self.keys is None:
            if self.levels is not None:
                raise ValueError('levels supported only when keys is not None')
            concat_axis = _concat_indexes(indexes)
        else:
            concat_axis = _make_concat_multiindex(indexes, self.keys, self.levels, self.names)
        self._maybe_check_integrity(concat_axis)
        return concat_axis

    def _maybe_check_integrity(self, concat_index: Index):
        if self.verify_integrity:
            if not concat_index.is_unique:
                overlap = concat_index[concat_index.duplicated()].unique()
                raise ValueError(f'Indexes have overlapping values: {overlap}')