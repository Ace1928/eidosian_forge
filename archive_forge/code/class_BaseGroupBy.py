from __future__ import annotations
from collections.abc import (
import datetime
from functools import (
import inspect
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config.config import option_context
from pandas._libs import (
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core._numba import executor
from pandas.core.apply import warn_alias_replacement
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
from pandas.core.indexes.api import (
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {'as_index', 'axis', 'dropna', 'exclusions', 'grouper', 'group_keys', 'keys', 'level', 'obj', 'observed', 'sort'}
    axis: AxisInt
    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None = None
    level: IndexLabel | None = None
    group_keys: bool

    @final
    def __len__(self) -> int:
        return len(self.groups)

    @final
    def __repr__(self) -> str:
        return object.__repr__(self)

    @final
    @property
    def grouper(self) -> ops.BaseGrouper:
        warnings.warn(f'{type(self).__name__}.grouper is deprecated and will be removed in a future version of pandas.', category=FutureWarning, stacklevel=find_stack_level())
        return self._grouper

    @final
    @property
    def groups(self) -> dict[Hashable, np.ndarray]:
        """
        Dict {group name -> group labels}.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).groups
        {'a': ['a', 'a'], 'b': ['b']}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> df.groupby(by=["a"]).groups
        {1: [0, 1], 7: [2]}

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').groups
        {Timestamp('2023-01-01 00:00:00'): 2, Timestamp('2023-02-01 00:00:00'): 4}
        """
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Dict {group name -> group indices}.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).indices
        {'a': array([0, 1]), 'b': array([2])}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).indices
        {1: array([0, 1]), 7: array([2])}

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').indices
        defaultdict(<class 'list'>, {Timestamp('2023-01-01 00:00:00'): [0, 1],
        Timestamp('2023-02-01 00:00:00'): [2, 3]})
        """
        return self._grouper.indices

    @final
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """

        def get_converter(s):
            if isinstance(s, datetime.datetime):
                return lambda key: Timestamp(key)
            elif isinstance(s, np.datetime64):
                return lambda key: Timestamp(key).asm8
            else:
                return lambda key: key
        if len(names) == 0:
            return []
        if len(self.indices) > 0:
            index_sample = next(iter(self.indices))
        else:
            index_sample = None
        name_sample = names[0]
        if isinstance(index_sample, tuple):
            if not isinstance(name_sample, tuple):
                msg = 'must supply a tuple to get_group with multiple grouping keys'
                raise ValueError(msg)
            if not len(name_sample) == len(index_sample):
                try:
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    msg = 'must supply a same-length tuple to get_group with multiple grouping keys'
                    raise ValueError(msg) from err
            converters = [get_converter(s) for s in index_sample]
            names = (tuple((f(n) for f, n in zip(converters, name))) for name in names)
        else:
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)
        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        if isinstance(self.obj, Series):
            return self.obj
        if self._selection is not None:
            if is_hashable(self._selection):
                return self.obj[self._selection]
            return self._obj_with_exclusions
        return self.obj

    @final
    def _dir_additions(self) -> set[str]:
        return self.obj._dir_additions()

    @Substitution(klass='GroupBy', examples=dedent("        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})\n        >>> df\n           A  B\n        0  a  1\n        1  b  2\n        2  a  3\n        3  b  4\n\n        To get the difference between each groups maximum and minimum value in one\n        pass, you can do\n\n        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())\n           B\n        A\n        a  2\n        b  2"))
    @Appender(_pipe_template)
    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T:
        return com.pipe(self, func, *args, **kwargs)

    @final
    def get_group(self, name, obj=None) -> DataFrame | Series:
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

            .. deprecated:: 2.1.0
                The obj is deprecated and will be removed in a future version.
                Do ``df.iloc[gb.indices.get(name)]``
                instead of ``gb.get_group(name, obj=df)``.

        Returns
        -------
        same type as obj

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).get_group("a")
        a    1
        a    2
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).get_group((1,))
                a  b  c
        owl     1  2  3
        toucan  1  5  6

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').get_group('2023-01-01')
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        """
        keys = self.keys
        level = self.level
        if is_list_like(level) and len(level) == 1 or (is_list_like(keys) and len(keys) == 1):
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            elif not isinstance(name, tuple):
                warnings.warn('When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.', FutureWarning, stacklevel=find_stack_level())
        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)
        if obj is None:
            indexer = inds if self.axis == 0 else (slice(None), inds)
            return self._selected_obj.iloc[indexer]
        else:
            warnings.warn('obj is deprecated and will be removed in a future version. Do ``df.iloc[gb.indices.get(name)]`` instead of ``gb.get_group(name, obj=df)``.', FutureWarning, stacklevel=find_stack_level())
            return obj._take_with_is_copy(inds, axis=self.axis)

    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator.

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> for x, y in ser.groupby(level=0):
        ...     print(f'{x}\\n{y}\\n')
        a
        a    1
        a    2
        dtype: int64
        b
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> for x, y in df.groupby(by=["a"]):
        ...     print(f'{x}\\n{y}\\n')
        (1,)
           a  b  c
        0  1  2  3
        1  1  5  6
        (7,)
           a  b  c
        2  7  8  9

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> for x, y in ser.resample('MS'):
        ...     print(f'{x}\\n{y}\\n')
        2023-01-01 00:00:00
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        2023-02-01 00:00:00
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        """
        keys = self.keys
        level = self.level
        result = self._grouper.get_iterator(self._selected_obj, axis=self.axis)
        if is_list_like(level) and len(level) == 1:
            warnings.warn('Creating a Groupby object with a length-1 list-like level parameter will yield indexes as tuples in a future version. To keep indexes as scalars, create Groupby objects with a scalar level parameter instead.', FutureWarning, stacklevel=find_stack_level())
        if isinstance(keys, list) and len(keys) == 1:
            result = (((key,), group) for key, group in result)
        return result