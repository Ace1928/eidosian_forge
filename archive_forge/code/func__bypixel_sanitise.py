from __future__ import annotations
from numbers import Number
from math import log10
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from xarray import DataArray, Dataset
from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd
def _bypixel_sanitise(source, glyph, agg):
    if isinstance(source, DataArray) and source.ndim == 1:
        if not source.name:
            source.name = 'value'
        source = source.reset_coords()
    if isinstance(source, Dataset) and len(source.dims) == 1:
        columns = list(source.coords.keys()) + list(source.data_vars.keys())
        cols_to_keep = _cols_to_keep(columns, glyph, agg)
        source = source.drop_vars([col for col in columns if col not in cols_to_keep])
        source = source.to_dask_dataframe()
    if isinstance(source, pd.DataFrame) or (cudf and isinstance(source, cudf.DataFrame)):
        cols_to_keep = _cols_to_keep(source.columns, glyph, agg)
        if len(cols_to_keep) < len(source.columns):
            sindex = None
            from .glyphs.polygon import PolygonGeom
            if isinstance(glyph, PolygonGeom):
                sindex = getattr(source[glyph.geometry].array, '_sindex', None)
            source = source[cols_to_keep]
            if sindex is not None and getattr(source[glyph.geometry].array, '_sindex', None) is None:
                source[glyph.geometry].array._sindex = sindex
        dshape = dshape_from_pandas(source)
    elif isinstance(source, dd.DataFrame):
        dshape, source = dshape_from_dask(source)
    elif isinstance(source, Dataset):
        dshape = dshape_from_xarray_dataset(source)
    else:
        raise ValueError('source must be a pandas or dask DataFrame')
    return (source, dshape)