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
class area_aggregate(AggregationOperation):
    """
    Aggregates Area elements by filling the area between zero and
    the y-values if only one value dimension is defined and the area
    between the curves if two are provided.
    """

    def _process(self, element, key=None):
        x, y = element.dimensions()[:2]
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        default = None
        if not self.p.y_range:
            y0, y1 = element.range(1)
            if len(element.vdims) > 1:
                y0, _ = element.range(2)
            elif y0 >= 0:
                y0 = 0
            elif y1 <= 0:
                y1 = 0
            default = (y0, y1)
        ystack = element.vdims[1].name if len(element.vdims) > 1 else None
        info = self._get_sampling(element, x, y, ndim=2, default=default)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        df = PandasInterface.as_dframe(element)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        params = self._get_agg_params(element, x, y, agg_fn, (x0, y0, x1, y1))
        if width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)
        agg = cvs.area(df, x.name, y.name, agg_fn, axis=0, y_stack=ystack)
        if xtype == 'datetime':
            agg[x.name] = agg[x.name].astype('datetime64[ns]')
        return self.p.element_type(agg, **params)