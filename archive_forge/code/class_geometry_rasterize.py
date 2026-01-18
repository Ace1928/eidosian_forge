import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
class geometry_rasterize(LineAggregationOperation):
    """
    Rasterizes geometries by converting them to spatialpandas.
    """
    aggregator = param.ClassSelector(default=rd.mean(), class_=(rd.Reduction, rd.summary, str))

    @classmethod
    def _get_aggregator(cls, element, agg, add_field=True):
        if not (element.vdims or isinstance(agg, str)) and agg.column is None and (not isinstance(agg, (rd.count, rd.any))):
            return ds.count()
        return super()._get_aggregator(element, agg, add_field)

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        xdim, ydim = element.kdims
        info = self._get_sampling(element, xdim, ydim)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        x0, x1 = x_range
        y0, y1 = y_range
        params = self._get_agg_params(element, xdim, ydim, agg_fn, (x0, y0, x1, y1))
        if width == 0 or height == 0:
            return self._empty_agg(element, xdim, ydim, width, height, xs, ys, agg_fn, **params)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        if element._plot_id in self._precomputed:
            data, col = self._precomputed[element._plot_id]
        else:
            if 'spatialpandas' not in element.interface.datatype:
                element = element.clone(datatype=['spatialpandas'])
            data = element.data
            col = element.interface.geo_column(data)
        if self.p.precompute:
            self._precomputed[element._plot_id] = (data, col)
        if isinstance(agg_fn, ds.count_cat) and data[agg_fn.column].dtype.name != 'category':
            data[agg_fn.column] = data[agg_fn.column].astype('category')
        agg_kwargs = dict(geometry=col, agg=agg_fn)
        if isinstance(element, Polygons):
            agg = cvs.polygons(data, **agg_kwargs)
        elif isinstance(element, Path):
            if self.p.line_width and ds_version >= Version('0.14.0'):
                agg_kwargs['line_width'] = self.p.line_width
            agg = cvs.line(data, **agg_kwargs)
        elif isinstance(element, Points):
            agg = cvs.points(data, **agg_kwargs)
        rename_dict = {k: v for k, v in zip('xy', (xdim.name, ydim.name)) if k != v}
        agg = agg.rename(rename_dict)
        if agg.ndim == 2:
            return self.p.element_type(agg, **params)
        else:
            layers = {}
            for c in agg.coords[agg_fn.column].data:
                cagg = agg.sel(**{agg_fn.column: c})
                layers[c] = self.p.element_type(cagg, **params)
            return NdOverlay(layers, kdims=[element.get_dimension(agg_fn.column)])