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
class quadmesh_rasterize(trimesh_rasterize):
    """
    Rasterize the QuadMesh element using the supplied aggregator.
    Simply converts to a TriMesh and lets trimesh_rasterize
    handle the actual rasterization.
    """

    def _precompute(self, element, agg):
        if ds_version <= Version('0.7.0'):
            return super()._precompute(element.trimesh(), agg)

    def _process(self, element, key=None):
        if ds_version <= Version('0.7.0'):
            return super()._process(element, key)
        if element.interface.datatype != 'xarray':
            element = element.clone(datatype=['xarray'])
        data = element.data
        x, y = element.kdims
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        if xtype == 'datetime':
            data[x.name] = data[x.name].astype('datetime64[ns]').astype('int64')
        if ytype == 'datetime':
            data[y.name] = data[y.name].astype('datetime64[ns]').astype('int64')
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        params = dict(get_param_values(element), datatype=['xarray'], bounds=(x0, y0, x1, y1))
        if width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        vdim = getattr(agg_fn, 'column', element.vdims[0].name)
        agg = cvs.quadmesh(data[vdim], x.name, y.name, agg_fn)
        xdim, ydim = list(agg.dims)[:2][::-1]
        if xtype == 'datetime':
            agg[xdim] = agg[xdim].astype('datetime64[ns]')
        if ytype == 'datetime':
            agg[ydim] = agg[ydim].astype('datetime64[ns]')
        return Image(agg, **params)