from __future__ import annotations
import functools
import itertools
import math
import warnings
from collections.abc import Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
import numpy as np
from packaging.version import Version
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.arithmetic import CoarsenArithmetic
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import CoarsenBoundaryOptions, SideOptions, T_Xarray
from xarray.core.utils import (
from xarray.namedarray import pycompat
class DataArrayCoarsen(Coarsen['DataArray']):
    __slots__ = ()
    _reduce_extra_args_docstring = ''

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool=False, numeric_only: bool=False) -> Callable[..., DataArray]:
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        kwargs: dict[str, Any] = {}
        if include_skipna:
            kwargs['skipna'] = None

        def wrapped_func(self: DataArrayCoarsen, keep_attrs: bool | None=None, **kwargs) -> DataArray:
            from xarray.core.dataarray import DataArray
            keep_attrs = self._get_keep_attrs(keep_attrs)
            reduced = self.obj.variable.coarsen(self.windows, func, self.boundary, self.side, keep_attrs, **kwargs)
            coords = {}
            for c, v in self.obj.coords.items():
                if c == self.obj.name:
                    coords[c] = reduced
                elif any((d in self.windows for d in v.dims)):
                    coords[c] = v.variable.coarsen(self.windows, self.coord_func[c], self.boundary, self.side, keep_attrs, **kwargs)
                else:
                    coords[c] = v
            return DataArray(reduced, dims=self.obj.dims, coords=coords, name=self.obj.name)
        return wrapped_func

    def reduce(self, func: Callable, keep_attrs: bool | None=None, **kwargs) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, axis, **kwargs)`
            to return the result of collapsing an np.ndarray over the coarsening
            dimensions.  It must be possible to provide the `axis` argument
            with a tuple of integers.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))
        >>> coarsen = da.coarsen(b=2)
        >>> coarsen.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 2)> Size: 32B
        array([[ 1,  5],
               [ 9, 13]])
        Dimensions without coordinates: a, b
        """
        wrapped_func = self._reduce_method(func)
        return wrapped_func(self, keep_attrs=keep_attrs, **kwargs)