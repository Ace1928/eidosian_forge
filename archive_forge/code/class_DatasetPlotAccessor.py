from __future__ import annotations
import functools
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Literal, NoReturn, overload
import numpy as np
from xarray.plot import dataarray_plot, dataset_plot
class DatasetPlotAccessor:
    """
    Enables use of xarray.plot functions as attributes on a Dataset.
    For example, Dataset.plot.scatter
    """
    _ds: Dataset
    __slots__ = ('_ds',)

    def __init__(self, dataset: Dataset) -> None:
        self._ds = dataset

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError('Dataset.plot cannot be called directly. Use an explicit plot method, e.g. ds.plot.scatter(...)')

    @overload
    def scatter(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend=None, levels=None, **kwargs: Any) -> PathCollection:
        ...

    @overload
    def scatter(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend=None, levels=None, **kwargs: Any) -> FacetGrid[Dataset]:
        ...

    @overload
    def scatter(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend=None, levels=None, **kwargs: Any) -> FacetGrid[Dataset]:
        ...

    @functools.wraps(dataset_plot.scatter, assigned=('__doc__',))
    def scatter(self, *args, **kwargs) -> PathCollection | FacetGrid[Dataset]:
        return dataset_plot.scatter(self._ds, *args, **kwargs)

    @overload
    def quiver(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, col: None=None, row: None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: AspectOptions=None, subplot_kws: dict[str, Any] | None=None, add_guide: bool | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals=None, center=None, levels=None, robust: bool | None=None, colors=None, extend=None, cmap=None, **kwargs: Any) -> Quiver:
        ...

    @overload
    def quiver(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, col: Hashable, row: Hashable | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: AspectOptions=None, subplot_kws: dict[str, Any] | None=None, add_guide: bool | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals=None, center=None, levels=None, robust: bool | None=None, colors=None, extend=None, cmap=None, **kwargs: Any) -> FacetGrid[Dataset]:
        ...

    @overload
    def quiver(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, col: Hashable | None=None, row: Hashable, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: AspectOptions=None, subplot_kws: dict[str, Any] | None=None, add_guide: bool | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals=None, center=None, levels=None, robust: bool | None=None, colors=None, extend=None, cmap=None, **kwargs: Any) -> FacetGrid[Dataset]:
        ...

    @functools.wraps(dataset_plot.quiver, assigned=('__doc__',))
    def quiver(self, *args, **kwargs) -> Quiver | FacetGrid[Dataset]:
        return dataset_plot.quiver(self._ds, *args, **kwargs)

    @overload
    def streamplot(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, col: None=None, row: None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: AspectOptions=None, subplot_kws: dict[str, Any] | None=None, add_guide: bool | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals=None, center=None, levels=None, robust: bool | None=None, colors=None, extend=None, cmap=None, **kwargs: Any) -> LineCollection:
        ...

    @overload
    def streamplot(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, col: Hashable, row: Hashable | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: AspectOptions=None, subplot_kws: dict[str, Any] | None=None, add_guide: bool | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals=None, center=None, levels=None, robust: bool | None=None, colors=None, extend=None, cmap=None, **kwargs: Any) -> FacetGrid[Dataset]:
        ...

    @overload
    def streamplot(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, u: Hashable | None=None, v: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, col: Hashable | None=None, row: Hashable, ax: Axes | None=None, figsize: Iterable[float] | None=None, size: float | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: AspectOptions=None, subplot_kws: dict[str, Any] | None=None, add_guide: bool | None=None, cbar_kwargs: dict[str, Any] | None=None, cbar_ax: Axes | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, infer_intervals=None, center=None, levels=None, robust: bool | None=None, colors=None, extend=None, cmap=None, **kwargs: Any) -> FacetGrid[Dataset]:
        ...

    @functools.wraps(dataset_plot.streamplot, assigned=('__doc__',))
    def streamplot(self, *args, **kwargs) -> LineCollection | FacetGrid[Dataset]:
        return dataset_plot.streamplot(self._ds, *args, **kwargs)