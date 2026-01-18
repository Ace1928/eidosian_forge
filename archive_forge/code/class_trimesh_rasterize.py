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
class trimesh_rasterize(aggregate):
    """
    Rasterize the TriMesh element using the supplied aggregator. If
    the TriMesh nodes or edges define a value dimension, will plot
    filled and shaded polygons; otherwise returns a wiremesh of the
    data.
    """
    aggregator = param.ClassSelector(default=rd.mean(), class_=(rd.Reduction, rd.summary, str))
    interpolation = param.ObjectSelector(default='bilinear', objects=['bilinear', 'linear', None, False], doc='\n        The interpolation method to apply during rasterization.')

    def _precompute(self, element, agg):
        from datashader.utils import mesh
        if element.vdims and getattr(agg, 'column', None) not in element.nodes.vdims:
            simplex_dims = [0, 1, 2, 3]
            vert_dims = [0, 1]
        elif element.nodes.vdims:
            simplex_dims = [0, 1, 2]
            vert_dims = [0, 1, 3]
        else:
            raise ValueError('Cannot shade TriMesh without value dimension.')
        datatypes = [element.interface.datatype, element.nodes.interface.datatype]
        if set(datatypes) == {'dask'}:
            dims, node_dims = (element.dimensions(), element.nodes.dimensions())
            simplices = element.data[[dims[sd].name for sd in simplex_dims]]
            verts = element.nodes.data[[node_dims[vd].name for vd in vert_dims]]
        else:
            if 'dask' in datatypes:
                if datatypes[0] == 'dask':
                    p, n = ('simplexes', 'vertices')
                else:
                    p, n = ('vertices', 'simplexes')
                self.param.warning(f'TriMesh {p} were provided as dask DataFrame but {n} were not. Datashader will not use dask to parallelize rasterization unless both are provided as dask DataFrames.')
            simplices = element.dframe(simplex_dims)
            verts = element.nodes.dframe(vert_dims)
        for c, dtype in zip(simplices.columns[:3], simplices.dtypes):
            if dtype.kind != 'i':
                simplices[c] = simplices[c].astype('int')
        mesh = mesh(verts, simplices)
        if hasattr(mesh, 'persist'):
            mesh = mesh.persist()
        return {'mesh': mesh, 'simplices': simplices, 'vertices': verts}

    def _precompute_wireframe(self, element, agg):
        if hasattr(element, '_wireframe'):
            segments = element._wireframe.data
        else:
            segments = connect_tri_edges_pd(element)
            element._wireframe = Dataset(segments, datatype=['dataframe', 'dask'])
        return {'segments': segments}

    def _process(self, element, key=None):
        if isinstance(element, TriMesh):
            x, y = element.nodes.kdims[:2]
        else:
            x, y = element.kdims
        info = self._get_sampling(element, x, y)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        agg = self.p.aggregator
        interp = self.p.interpolation or None
        precompute = self.p.precompute
        if interp == 'linear':
            interp = 'bilinear'
        wireframe = False
        if not (element.vdims or (isinstance(element, TriMesh) and element.nodes.vdims)) and ds_version <= Version('0.6.9'):
            self.p.aggregator = ds.any() if isinstance(agg, ds.any) or agg == 'any' else ds.count()
            return aggregate._process(self, element, key)
        elif not interp and (isinstance(agg, (ds.any, ds.count)) or agg in ['any', 'count']) or not (element.vdims or element.nodes.vdims):
            wireframe = True
            precompute = False
            if isinstance(agg, (ds.any, ds.count)):
                agg = self._get_aggregator(element, self.p.aggregator)
            else:
                agg = ds.any()
        elif getattr(agg, 'column', None) is None:
            agg = self._get_aggregator(element, self.p.aggregator)
        if element._plot_id in self._precomputed:
            precomputed = self._precomputed[element._plot_id]
        elif wireframe:
            precomputed = self._precompute_wireframe(element, agg)
        else:
            precomputed = self._precompute(element, agg)
        bounds = (x_range[0], y_range[0], x_range[1], y_range[1])
        params = self._get_agg_params(element, x, y, agg, bounds)
        if width == 0 or height == 0:
            if width == 0:
                params['xdensity'] = 1
            if height == 0:
                params['ydensity'] = 1
            return Image((xs, ys, np.zeros((height, width))), **params)
        if wireframe:
            segments = precomputed['segments']
        else:
            simplices = precomputed['simplices']
            pts = precomputed['vertices']
            mesh = precomputed['mesh']
        if precompute:
            self._precomputed = {element._plot_id: precomputed}
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        if wireframe:
            rename_dict = {k: v for k, v in zip('xy', (x.name, y.name)) if k != v}
            agg = cvs.line(segments, x=['x0', 'x1', 'x2', 'x0'], y=['y0', 'y1', 'y2', 'y0'], axis=1, agg=agg).rename(rename_dict)
        else:
            interpolate = bool(self.p.interpolation)
            agg = cvs.trimesh(pts, simplices, agg=agg, interp=interpolate, mesh=mesh)
        return Image(agg, **params)