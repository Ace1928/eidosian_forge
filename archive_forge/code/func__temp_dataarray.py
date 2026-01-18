from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from xarray.core.alignment import broadcast
from xarray.plot import dataarray_plot
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
def _temp_dataarray(ds: Dataset, y: Hashable, locals_: dict[str, Any]) -> DataArray:
    """Create a temporary datarray with extra coords."""
    from xarray.core.dataarray import DataArray
    coords = dict(ds.coords)
    valid_coord_kwargs = {'x', 'z', 'markersize', 'hue', 'row', 'col', 'u', 'v'}
    coord_kwargs = locals_.keys() & valid_coord_kwargs
    for k in coord_kwargs:
        key = locals_[k]
        if ds.data_vars.get(key) is not None:
            coords[key] = ds[key]
    _y = ds[y].broadcast_like(ds)
    return DataArray(_y, coords=coords)