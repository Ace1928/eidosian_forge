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
def _update_doc_to_dataset(dataarray_plotfunc: Callable) -> Callable[[F], F]:
    """
    Add a common docstring by re-using the DataArray one.

    TODO: Reduce code duplication.

    * The goal is to reduce code duplication by moving all Dataset
      specific plots to the DataArray side and use this thin wrapper to
      handle the conversion between Dataset and DataArray.
    * Improve docstring handling, maybe reword the DataArray versions to
      explain Datasets better.

    Parameters
    ----------
    dataarray_plotfunc : Callable
        Function that returns a finished plot primitive.
    """
    da_doc = dataarray_plotfunc.__doc__
    if da_doc is None:
        raise NotImplementedError('DataArray plot method requires a docstring')
    da_str = '\n    Parameters\n    ----------\n    darray : DataArray\n    '
    ds_str = '\n\n    The `y` DataArray will be used as base, any other variables are added as coords.\n\n    Parameters\n    ----------\n    ds : Dataset\n    '
    if da_str in da_doc:
        ds_doc = da_doc.replace(da_str, ds_str).replace('darray', 'ds')
    else:
        ds_doc = da_doc

    @functools.wraps(dataarray_plotfunc)
    def wrapper(dataset_plotfunc: F) -> F:
        dataset_plotfunc.__doc__ = ds_doc
        return dataset_plotfunc
    return wrapper