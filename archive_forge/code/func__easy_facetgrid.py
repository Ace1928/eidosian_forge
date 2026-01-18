from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def _easy_facetgrid(data: T_DataArrayOrSet, plotfunc: Callable, kind: Literal['line', 'dataarray', 'dataset', 'plot1d'], x: Hashable | None=None, y: Hashable | None=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: float | None=None, size: float | None=None, subplot_kws: dict[str, Any] | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, **kwargs: Any) -> FacetGrid[T_DataArrayOrSet]:
    """
    Convenience method to call xarray.plot.FacetGrid from 2d plotting methods

    kwargs are the arguments to 2d plotting method
    """
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError('cannot provide both `figsize` and `size` arguments')
    if kwargs.get('z') is not None:
        sharex = False
        sharey = False
    g = FacetGrid(data=data, col=col, row=row, col_wrap=col_wrap, sharex=sharex, sharey=sharey, figsize=figsize, aspect=aspect, size=size, subplot_kws=subplot_kws)
    if kind == 'line':
        return g.map_dataarray_line(plotfunc, x, y, **kwargs)
    if kind == 'dataarray':
        return g.map_dataarray(plotfunc, x, y, **kwargs)
    if kind == 'plot1d':
        return g.map_plot1d(plotfunc, x, y, **kwargs)
    if kind == 'dataset':
        return g.map_dataset(plotfunc, x, y, **kwargs)
    raise ValueError(f'kind must be one of `line`, `dataarray`, `dataset` or `plot1d`, got {kind}')