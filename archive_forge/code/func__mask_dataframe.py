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