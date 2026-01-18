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
def _empty_agg(self, element, x, y, width, height, xs, ys, agg_fn, **params):
    x = x.name if x else 'x'
    y = y.name if x else 'y'
    xarray = xr.DataArray(np.full((height, width), np.nan), dims=[y, x], coords={x: xs, y: ys})
    if width == 0:
        params['xdensity'] = 1
    if height == 0:
        params['ydensity'] = 1
    el = self.p.element_type(xarray, **params)
    if isinstance(agg_fn, ds.count_cat):
        vals = element.dimension_values(agg_fn.column, expanded=False)
        dim = element.get_dimension(agg_fn.column)
        return NdOverlay({v: el for v in vals}, dim)
    return el