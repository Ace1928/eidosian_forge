from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
class _Normalize(Sequence):
    """
    Normalize numerical or categorical values to numerical values.

    The class includes helper methods that simplifies transforming to
    and from normalized values.

    Parameters
    ----------
    data : DataArray
        DataArray to normalize.
    width : Sequence of three numbers, optional
        Normalize the data to these (min, default, max) values.
        The default is None.
    """
    _data: DataArray | None
    _data_unique: np.ndarray
    _data_unique_index: np.ndarray
    _data_unique_inverse: np.ndarray
    _data_is_numeric: bool
    _width: tuple[float, float, float] | None
    __slots__ = ('_data', '_data_unique', '_data_unique_index', '_data_unique_inverse', '_data_is_numeric', '_width')

    def __init__(self, data: DataArray | None, width: tuple[float, float, float] | None=None, _is_facetgrid: bool=False) -> None:
        self._data = data
        self._width = width if not _is_facetgrid else None
        pint_array_type = DuckArrayModule('pint').type
        to_unique = data.to_numpy() if isinstance(data if data is None else data.data, pint_array_type) else data
        data_unique, data_unique_inverse = np.unique(to_unique, return_inverse=True)
        self._data_unique = data_unique
        self._data_unique_index = np.arange(0, data_unique.size)
        self._data_unique_inverse = data_unique_inverse
        self._data_is_numeric = False if data is None else _is_numeric(data)

    def __repr__(self) -> str:
        with np.printoptions(precision=4, suppress=True, threshold=5):
            return f'<_Normalize(data, width={self._width})>\n{self._data_unique} -> {self._values_unique}'

    def __len__(self) -> int:
        return len(self._data_unique)

    def __getitem__(self, key):
        return self._data_unique[key]

    @property
    def data(self) -> DataArray | None:
        return self._data

    @property
    def data_is_numeric(self) -> bool:
        """
        Check if data is numeric.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).data_is_numeric
        False

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a).data_is_numeric
        True

        >>> # TODO: Datetime should be numeric right?
        >>> a = xr.DataArray(pd.date_range("2000-1-1", periods=4))
        >>> _Normalize(a).data_is_numeric
        False

        # TODO: Timedelta should be numeric right?
        >>> a = xr.DataArray(pd.timedelta_range("-1D", periods=4, freq="D"))
        >>> _Normalize(a).data_is_numeric
        True
        """
        return self._data_is_numeric

    @overload
    def _calc_widths(self, y: np.ndarray) -> np.ndarray:
        ...

    @overload
    def _calc_widths(self, y: DataArray) -> DataArray:
        ...

    def _calc_widths(self, y: np.ndarray | DataArray) -> np.ndarray | DataArray:
        """
        Normalize the values so they're in between self._width.
        """
        if self._width is None:
            return y
        xmin, xdefault, xmax = self._width
        diff_maxy_miny = np.max(y) - np.min(y)
        if diff_maxy_miny == 0:
            widths = xdefault + 0 * y
        else:
            k = (y - np.min(y)) / diff_maxy_miny
            widths = xmin + k * (xmax - xmin)
        return widths

    @overload
    def _indexes_centered(self, x: np.ndarray) -> np.ndarray:
        ...

    @overload
    def _indexes_centered(self, x: DataArray) -> DataArray:
        ...

    def _indexes_centered(self, x: np.ndarray | DataArray) -> np.ndarray | DataArray:
        """
        Offset indexes to make sure being in the center of self.levels.
        ["a", "b", "c"] -> [1, 3, 5]
        """
        return x * 2 + 1

    @property
    def values(self) -> DataArray | None:
        """
        Return a normalized number array for the unique levels.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).values
        <xarray.DataArray (dim_0: 5)> Size: 40B
        array([3, 1, 1, 3, 5])
        Dimensions without coordinates: dim_0

        >>> _Normalize(a, width=(18, 36, 72)).values
        <xarray.DataArray (dim_0: 5)> Size: 40B
        array([45., 18., 18., 45., 72.])
        Dimensions without coordinates: dim_0

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a).values
        <xarray.DataArray (dim_0: 6)> Size: 48B
        array([0.5, 0. , 0. , 0.5, 2. , 3. ])
        Dimensions without coordinates: dim_0

        >>> _Normalize(a, width=(18, 36, 72)).values
        <xarray.DataArray (dim_0: 6)> Size: 48B
        array([27., 18., 18., 27., 54., 72.])
        Dimensions without coordinates: dim_0

        >>> _Normalize(a * 0, width=(18, 36, 72)).values
        <xarray.DataArray (dim_0: 6)> Size: 48B
        array([36., 36., 36., 36., 36., 36.])
        Dimensions without coordinates: dim_0

        """
        if self.data is None:
            return None
        val: DataArray
        if self.data_is_numeric:
            val = self.data
        else:
            arr = self._indexes_centered(self._data_unique_inverse)
            val = self.data.copy(data=arr.reshape(self.data.shape))
        return self._calc_widths(val)

    @property
    def _values_unique(self) -> np.ndarray | None:
        """
        Return unique values.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a)._values_unique
        array([1, 3, 5])

        >>> _Normalize(a, width=(18, 36, 72))._values_unique
        array([18., 45., 72.])

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a)._values_unique
        array([0. , 0.5, 2. , 3. ])

        >>> _Normalize(a, width=(18, 36, 72))._values_unique
        array([18., 27., 54., 72.])
        """
        if self.data is None:
            return None
        val: np.ndarray
        if self.data_is_numeric:
            val = self._data_unique
        else:
            val = self._indexes_centered(self._data_unique_index)
        return self._calc_widths(val)

    @property
    def ticks(self) -> np.ndarray | None:
        """
        Return ticks for plt.colorbar if the data is not numeric.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).ticks
        array([1, 3, 5])
        """
        val: None | np.ndarray
        if self.data_is_numeric:
            val = None
        else:
            val = self._indexes_centered(self._data_unique_index)
        return val

    @property
    def levels(self) -> np.ndarray:
        """
        Return discrete levels that will evenly bound self.values.
        ["a", "b", "c"] -> [0, 2, 4, 6]

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).levels
        array([0, 2, 4, 6])
        """
        return np.append(self._data_unique_index, np.max(self._data_unique_index) + 1) * 2

    @property
    def _lookup(self) -> pd.Series:
        if self._values_unique is None:
            raise ValueError("self.data can't be None.")
        return pd.Series(dict(zip(self._values_unique, self._data_unique)))

    def _lookup_arr(self, x) -> np.ndarray:
        return self._lookup.sort_index().reindex(x, method='nearest').to_numpy()

    @property
    def format(self) -> FuncFormatter:
        """
        Return a FuncFormatter that maps self.values elements back to
        the original value as a string. Useful with plt.colorbar.

        Examples
        --------
        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> aa = _Normalize(a, width=(0, 0.5, 1))
        >>> aa._lookup
        0.000000    0.0
        0.166667    0.5
        0.666667    2.0
        1.000000    3.0
        dtype: float64
        >>> aa.format(1)
        '3.0'
        """
        import matplotlib.pyplot as plt

        def _func(x: Any, pos: None | Any=None):
            return f'{self._lookup_arr([x])[0]}'
        return plt.FuncFormatter(_func)

    @property
    def func(self) -> Callable[[Any, None | Any], Any]:
        """
        Return a lambda function that maps self.values elements back to
        the original value as a numpy array. Useful with ax.legend_elements.

        Examples
        --------
        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> aa = _Normalize(a, width=(0, 0.5, 1))
        >>> aa._lookup
        0.000000    0.0
        0.166667    0.5
        0.666667    2.0
        1.000000    3.0
        dtype: float64
        >>> aa.func([0.16, 1])
        array([0.5, 3. ])
        """

        def _func(x: Any, pos: None | Any=None):
            return self._lookup_arr(x)
        return _func