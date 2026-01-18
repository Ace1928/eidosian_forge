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
class rasterize(AggregationOperation):
    """
    Rasterize is a high-level operation that will rasterize any
    Element or combination of Elements, aggregating them with the supplied
    aggregator and interpolation method.

    The default aggregation method depends on the type of Element but
    usually defaults to the count of samples in each bin. Other
    aggregators can be supplied implementing mean, max, min and other
    reduction operations.

    The bins of the aggregate are defined by the width and height and
    the x_range and y_range. If x_sampling or y_sampling are supplied
    the operation will ensure that a bin is no smaller than the minimum
    sampling distance by reducing the width and height when zoomed in
    beyond the minimum sampling distance.

    By default, the PlotSize and RangeXY streams are applied when this
    operation is used dynamically, which means that the width, height,
    x_range and y_range will automatically be set to match the inner
    dimensions of the linked plot and the ranges of the axes.
    """
    aggregator = param.ClassSelector(class_=(rd.Reduction, rd.summary, str), default='default')
    interpolation = param.ObjectSelector(default='default', objects=['default', 'linear', 'nearest', 'bilinear', None, False], doc='\n        The interpolation method to apply during rasterization.\n        Default depends on element type')
    _transforms = [(Image, regrid), (Polygons, geometry_rasterize), (lambda x: isinstance(x, (Path, Points)) and 'spatialpandas' in x.interface.datatype, geometry_rasterize), (TriMesh, trimesh_rasterize), (QuadMesh, quadmesh_rasterize), (lambda x: isinstance(x, NdOverlay) and issubclass(x.type, (Scatter, Points, Curve, Path)), aggregate), (Spikes, spikes_aggregate), (Area, area_aggregate), (Spread, spread_aggregate), (Segments, segments_aggregate), (Rectangles, rectangle_aggregate), (Contours, contours_rasterize), (Graph, aggregate), (Scatter, aggregate), (Points, aggregate), (Curve, aggregate), (Path, aggregate), (type(None), shade)]
    __instance_params = set()
    __instance_kwargs = {}

    @bothmethod
    def instance(self_or_cls, **params):
        kwargs = set(params) - set(self_or_cls.param)
        inst_params = {k: v for k, v in params.items() if k in self_or_cls.param}
        inst = super().instance(**inst_params)
        inst.__instance_params = set(inst_params)
        inst.__instance_kwargs = {k: v for k, v in params.items() if k in kwargs}
        return inst

    def _process(self, element, key=None):
        all_allowed_kws = set()
        all_supplied_kws = set()
        instance_params = dict(self.__instance_kwargs, **{k: getattr(self, k) for k in self.__instance_params})
        for predicate, transform in self._transforms:
            merged_param_values = dict(instance_params, **self.p)
            for k in ['aggregator', 'interpolation']:
                if merged_param_values.get(k, None) == 'default':
                    merged_param_values.pop(k)
            op_params = dict({k: v for k, v in merged_param_values.items() if not (v is None and k == 'aggregator')}, dynamic=False)
            extended_kws = dict(op_params, **self.p.extra_keywords())
            all_supplied_kws |= set(extended_kws)
            all_allowed_kws |= set(transform.param)
            op = transform.instance(**{k: v for k, v in extended_kws.items() if k in transform.param})
            op._precomputed = self._precomputed
            element = element.map(op, predicate)
            self._precomputed = op._precomputed
        unused_params = list(all_supplied_kws - all_allowed_kws)
        if unused_params:
            self.param.warning('Parameter(s) [%s] not consumed by any element rasterizer.' % ', '.join(unused_params))
        return element