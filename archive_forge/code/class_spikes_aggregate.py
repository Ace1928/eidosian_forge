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
class spikes_aggregate(LineAggregationOperation):
    """
    Aggregates Spikes elements by drawing individual line segments
    over the entire y_range if no value dimension is defined and
    between zero and the y-value if one is defined.
    """
    spike_length = param.Number(default=None, allow_None=True, doc='\n      If numeric, specifies the length of each spike, overriding the\n      vdims values (if present).')
    offset = param.Number(default=0.0, doc='\n      The offset of the lower end of each spike.')

    def _process(self, element, key=None):
        agg_fn = self._get_aggregator(element, self.p.aggregator)
        x, y = (element.kdims[0], None)
        spike_length = 0.5 if self.p.spike_length is None else self.p.spike_length
        if element.vdims and self.p.spike_length is None:
            x, y = element.dimensions()[:2]
            rename_dict = {'x': x.name, 'y': y.name}
            if not self.p.y_range:
                y0, y1 = element.range(1)
                if y0 >= 0:
                    default = (0, y1)
                elif y1 <= 0:
                    default = (y0, 0)
                else:
                    default = (y0, y1)
            else:
                default = None
        else:
            x, y = (element.kdims[0], None)
            default = (float(self.p.offset), float(self.p.offset + spike_length))
            rename_dict = {'x': x.name}
        info = self._get_sampling(element, x, y, ndim=1, default=default)
        (x_range, y_range), (xs, ys), (width, height), (xtype, ytype) = info
        ((x0, x1), (y0, y1)), (xs, ys) = self._dt_transform(x_range, y_range, xs, ys, xtype, ytype)
        value_cols = [] if agg_fn.column is None else [agg_fn.column]
        if y is None:
            df = element.dframe([x] + value_cols).copy()
            y = Dimension('y')
            df['y0'] = float(self.p.offset)
            df['y1'] = float(self.p.offset + spike_length)
            yagg = ['y0', 'y1']
            if not self.p.expand:
                height = 1
        else:
            df = element.dframe([x, y] + value_cols).copy()
            df['y0'] = np.array(0, df.dtypes[y.name])
            yagg = ['y0', y.name]
        if xtype == 'datetime':
            df[x.name] = cast_array_to_int64(df[x.name].astype('datetime64[ns]'))
        params = self._get_agg_params(element, x, y, agg_fn, (x0, y0, x1, y1))
        if width == 0 or height == 0:
            return self._empty_agg(element, x, y, width, height, xs, ys, agg_fn, **params)
        cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        agg_kwargs = {}
        if ds_version >= Version('0.14.0'):
            agg_kwargs['line_width'] = self.p.line_width
        rename_dict = {k: v for k, v in rename_dict.items() if k != v}
        agg = cvs.line(df, x.name, yagg, agg_fn, axis=1, **agg_kwargs).rename(rename_dict)
        if xtype == 'datetime':
            agg[x.name] = agg[x.name].astype('datetime64[ns]')
        return self.p.element_type(agg, **params)