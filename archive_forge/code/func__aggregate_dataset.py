import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def _aggregate_dataset(self, obj):
    """
        Generates a gridded Dataset from a column-based dataset and
        lists of xcoords and ycoords
        """
    xcoords, ycoords = self._get_coords(obj)
    dim_labels = obj.dimensions(label=True)
    vdims = obj.dimensions()[2:]
    xdim, ydim = dim_labels[:2]
    shape = (len(ycoords), len(xcoords))
    nsamples = np.prod(shape)
    grid_data = {xdim: xcoords, ydim: ycoords}
    ys, xs = cartesian_product([ycoords, xcoords], copy=True)
    data = {xdim: xs, ydim: ys}
    for vdim in vdims:
        values = np.empty(nsamples)
        values[:] = np.nan
        data[vdim.name] = values
    dtype = default_datatype
    dense_data = Dataset(data, kdims=obj.kdims, vdims=obj.vdims, datatype=[dtype])
    concat_data = obj.interface.concatenate([dense_data, obj], datatype=dtype)
    reindexed = concat_data.reindex([xdim, ydim], vdims)
    if not reindexed:
        agg = reindexed
    df = PandasInterface.as_dframe(reindexed)
    df = df.groupby([xdim, ydim], sort=False).first().reset_index()
    agg = reindexed.clone(df)
    for vdim in vdims:
        grid_data[vdim.name] = agg.dimension_values(vdim).reshape(shape)
    return agg.clone(grid_data, kdims=[xdim, ydim], vdims=vdims, datatype=self.p.datatype)