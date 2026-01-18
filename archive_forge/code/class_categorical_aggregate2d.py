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
class categorical_aggregate2d(Operation):
    """
    Generates a gridded Dataset of 2D aggregate arrays indexed by the
    first two dimensions of the passed Element, turning all remaining
    dimensions into value dimensions. The key dimensions of the
    gridded array are treated as categorical indices. Useful for data
    indexed by two independent categorical variables such as a table
    of population values indexed by country and year. Data that is
    indexed by continuous dimensions should be binned before
    aggregation. The aggregation will retain the global sorting order
    of both dimensions.

    >> table = Table([('USA', 2000, 282.2), ('UK', 2005, 58.89)],
                     kdims=['Country', 'Year'], vdims=['Population'])
    >> categorical_aggregate2d(table)
    Dataset({'Country': ['USA', 'UK'], 'Year': [2000, 2005],
             'Population': [[ 282.2 , np.nan], [np.nan,   58.89]]},
            kdims=['Country', 'Year'], vdims=['Population'])
    """
    datatype = param.List(default=['xarray', 'grid'], doc='\n        The grid interface types to use when constructing the gridded Dataset.')

    @classmethod
    def _get_coords(cls, obj):
        """
        Get the coordinates of the 2D aggregate, maintaining the correct
        sorting order.
        """
        xdim, ydim = obj.dimensions(label=True)[:2]
        xcoords = obj.dimension_values(xdim, False)
        ycoords = obj.dimension_values(ydim, False)
        if xcoords.dtype.kind not in 'SUO':
            xcoords = np.sort(xcoords)
        if ycoords.dtype.kind not in 'SUO':
            return (xcoords, np.sort(ycoords))
        grouped = obj.groupby(xdim, container_type=dict, group_type=Dataset).values()
        orderings = {}
        sort = True
        for group in grouped:
            vals = group.dimension_values(ydim, False)
            if len(vals) == 1:
                orderings[vals[0]] = [vals[0]]
            else:
                for i in range(len(vals) - 1):
                    p1, p2 = vals[i:i + 2]
                    orderings[p1] = [p2]
            if sort:
                if vals.dtype.kind in ('i', 'f'):
                    sort = (np.diff(vals) >= 0).all()
                else:
                    sort = np.array_equal(np.sort(vals), vals)
        if sort or one_to_one(orderings, ycoords):
            ycoords = np.sort(ycoords)
        elif not is_cyclic(orderings):
            coords = list(itertools.chain(*sort_topologically(orderings)))
            ycoords = coords if len(coords) == len(ycoords) else np.sort(ycoords)
        return (np.asarray(xcoords), np.asarray(ycoords))

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

    def _aggregate_dataset_pandas(self, obj):
        index_cols = [d.name for d in obj.kdims]
        df = obj.data.set_index(index_cols).groupby(index_cols, sort=False).first()
        label = 'unique' if len(df) == len(obj) else 'non-unique'
        levels = self._get_coords(obj)
        index = pd.MultiIndex.from_product(levels, names=df.index.names)
        reindexed = df.reindex(index)
        data = tuple(levels)
        shape = tuple((d.shape[0] for d in data))
        for vdim in obj.vdims:
            data += (reindexed[vdim.name].values.reshape(shape).T,)
        return obj.clone(data, datatype=self.p.datatype, label=label)

    def _process(self, obj, key=None):
        """
        Generates a categorical 2D aggregate by inserting NaNs at all
        cross-product locations that do not already have a value assigned.
        Returns a 2D gridded Dataset object.
        """
        if isinstance(obj, Dataset) and obj.interface.gridded:
            return obj
        elif obj.ndims > 2:
            raise ValueError('Cannot aggregate more than two dimensions')
        elif len(obj.dimensions()) < 3:
            raise ValueError('Must have at two dimensions to aggregate overand one value dimension to aggregate on.')
        obj = Dataset(obj, datatype=['dataframe'])
        return self._aggregate_dataset_pandas(obj)