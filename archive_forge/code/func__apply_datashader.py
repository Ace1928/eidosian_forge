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
def _apply_datashader(self, dfdata, cvs_fn, agg_fn, agg_kwargs, x, y):
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='casting datetime64', category=FutureWarning)
        agg = cvs_fn(dfdata, x.name, y.name, agg_fn, **agg_kwargs)
    is_where_index = ds15 and isinstance(agg_fn, ds.where) and isinstance(agg_fn.column, rd.SpecialColumn)
    is_summary_index = isinstance(agg_fn, ds.summary) and 'index' in agg
    if is_where_index or is_summary_index:
        if is_where_index:
            data = agg.data
            agg = agg.to_dataset(name='index')
        else:
            data = agg.index.data
        neg1 = data == -1
        for col in dfdata.columns:
            if col in agg.coords:
                continue
            val = dfdata[col].values[data]
            if val.dtype.kind == 'f':
                val[neg1] = np.nan
            elif isinstance(val.dtype, pd.CategoricalDtype):
                val = val.to_numpy()
                val[neg1] = '-'
            elif val.dtype.kind == 'O':
                val[neg1] = '-'
            elif val.dtype.kind == 'M':
                val[neg1] = np.datetime64('NaT')
            else:
                val = val.astype(np.float64)
                val[neg1] = np.nan
            agg[col] = ((y.name, x.name), val)
    return agg