import warnings
import numpy as np
import pandas as pd
from pandas.plotting import PlotAccessor
from pandas import CategoricalDtype
import geopandas
from packaging.version import Version
from ._decorator import doc
def _plot_polygon_collection(ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs.
    edgecolor : single color or sequence of `N` colors
        Color for the edge of the polygons
    facecolor : single color or sequence of `N` colors
        Color to fill the polygons. Cannot be used together with `values`.
    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`
    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import PatchCollection
    geoms, multiindex = _sanitize_geoms(geoms)
    if values is not None:
        values = np.take(values, multiindex, axis=0)
    kwargs = {att: value for att, value in kwargs.items() if att not in ['markersize', 'marker']}
    if color is not None:
        kwargs['color'] = color
    _expand_kwargs(kwargs, multiindex)
    collection = PatchCollection([_PolygonPatch(poly) for poly in geoms], **kwargs)
    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if 'norm' not in kwargs:
            collection.set_clim(vmin, vmax)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection