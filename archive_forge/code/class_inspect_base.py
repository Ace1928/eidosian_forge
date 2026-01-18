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
class inspect_base(inspect):
    """
    Given datashaded aggregate (Image) output, return a set of
    (hoverable) points sampled from those near the cursor.
    """

    def _process(self, raster, key=None):
        self._validate(raster)
        if isinstance(raster, RGB):
            raster = raster[..., raster.vdims[-1]]
        x_range, y_range = (raster.range(0), raster.range(1))
        xdelta, ydelta = self._distance_args(raster, x_range, y_range, self.p.pixels)
        x, y = (self.p.x, self.p.y)
        val = raster[x - xdelta:x + xdelta, y - ydelta:y + ydelta].reduce(function=np.nansum)
        if np.isnan(val):
            val = self.p.null_value
        if self.p.value_bounds and (not self.p.value_bounds[0] < val < self.p.value_bounds[1]) or val == self.p.null_value:
            result = self._empty_df(raster.dataset)
        else:
            masked = self._mask_dataframe(raster, x, y, xdelta, ydelta)
            result = self._sort_by_distance(raster, masked, x, y)
        self.hits = result
        df = self.p.transform(result)
        return self._element(raster, df.iloc[:self.p.max_indicators])

    @classmethod
    def _distance_args(cls, element, x_range, y_range, pixels):
        ycount, xcount = element.interface.shape(element, gridded=True)
        x_delta = abs(x_range[1] - x_range[0]) / xcount
        y_delta = abs(y_range[1] - y_range[0]) / ycount
        return (x_delta * pixels, y_delta * pixels)

    @classmethod
    def _empty_df(cls, dataset):
        if 'dask' in dataset.interface.datatype:
            return dataset.data._meta.iloc[:0]
        elif dataset.interface.datatype in ['pandas', 'geopandas', 'spatialpandas']:
            return dataset.data.head(0)
        return dataset.iloc[:0].dframe()

    @classmethod
    def _mask_dataframe(cls, raster, x, y, xdelta, ydelta):
        """
        Mask the dataframe around the specified x and y position with
        the given x and y deltas
        """
        ds = raster.dataset
        x0, x1, y0, y1 = (x - xdelta, x + xdelta, y - ydelta, y + ydelta)
        if 'spatialpandas' in ds.interface.datatype:
            df = ds.data.cx[x0:x1, y0:y1]
            return df.compute() if hasattr(df, 'compute') else df
        xdim, ydim = raster.kdims
        query = {xdim.name: (x0, x1), ydim.name: (y0, y1)}
        return ds.select(**query).dframe()

    @classmethod
    def _validate(cls, raster):
        pass

    @classmethod
    def _vdims(cls, raster, df):
        ds = raster.dataset
        if 'spatialpandas' in ds.interface.datatype:
            coords = [ds.interface.geo_column(ds.data)]
        else:
            coords = [kd.name for kd in raster.kdims]
        return [col for col in df.columns if col not in coords]