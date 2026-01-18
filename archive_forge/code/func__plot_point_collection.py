import warnings
import numpy as np
import pandas as pd
from pandas.plotting import PlotAccessor
from pandas import CategoricalDtype
import geopandas
from packaging.version import Version
from ._decorator import doc
def _plot_point_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, marker='o', markersize=None, **kwargs):
    """
    Plots a collection of Point and MultiPoint geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : sequence of `N` Points or MultiPoints

    values : a sequence of `N` values, optional
        Values mapped to colors using vmin, vmax, and cmap.
        Cannot be specified together with `color`.
    markersize : scalar or array-like, optional
        Size of the markers. Note that under the hood ``scatter`` is
        used, so the specified value will be proportional to the
        area of the marker (size in points^2).

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    if values is not None and color is not None:
        raise ValueError("Can only specify one of 'values' and 'color' kwargs")
    geoms, multiindex = _sanitize_geoms(geoms)
    x = [p.x if not p.is_empty else None for p in geoms]
    y = [p.y if not p.is_empty else None for p in geoms]
    if values is not None:
        kwargs['c'] = values
    if markersize is not None:
        kwargs['s'] = markersize
    if color is not None:
        kwargs['color'] = color
    if marker is not None:
        kwargs['marker'] = marker
    _expand_kwargs(kwargs, multiindex)
    if 'norm' not in kwargs:
        collection = ax.scatter(x, y, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    else:
        collection = ax.scatter(x, y, cmap=cmap, **kwargs)
    return collection