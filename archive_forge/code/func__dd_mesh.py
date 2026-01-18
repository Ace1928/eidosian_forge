from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def _dd_mesh(vertices, simplices):
    """Helper for ``datashader.utils.mesh()``. Both arguments are assumed to be
    Dask DataFrame objects.
    """
    res = _pd_mesh(vertices.compute(), simplices.compute())
    approx_npartitions = max(vertices.npartitions, simplices.npartitions)
    chunksize = int(np.ceil(len(res) / (3 * approx_npartitions)) * 3)
    res = dd.from_pandas(res, chunksize=chunksize)
    return res