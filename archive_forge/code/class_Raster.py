import colorsys
from copy import deepcopy
from operator import itemgetter
import numpy as np
import param
from ..core import Dataset, Dimension, Element2D, Overlay, config, util
from ..core.boundingregion import BoundingBox, BoundingRegion
from ..core.data import ImageInterface
from ..core.data.interface import DataError
from ..core.dimension import dimension_name
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from .chart import Curve
from .geom import Selection2DExpr
from .graphs import TriMesh
from .tabular import Table
from .util import categorical_aggregate2d, compute_slice_bounds
class Raster(Element2D):
    """
    Raster is a basic 2D element type for presenting either numpy or
    dask arrays as two dimensional raster images.

    Arrays with a shape of (N,M) are valid inputs for Raster whereas
    subclasses of Raster (e.g. RGB) may also accept 3D arrays
    containing channel information.

    Raster does not support slicing like the Image or RGB subclasses
    and the extents are in matrix coordinates if not explicitly
    specified.
    """
    kdims = param.List(default=[Dimension('x'), Dimension('y')], bounds=(2, 2), constant=True, doc='\n        The label of the x- and y-dimension of the Raster in form\n        of a string or dimension object.')
    group = param.String(default='Raster', constant=True)
    vdims = param.List(default=[Dimension('z')], bounds=(1, None), doc='\n        The dimension description of the data held in the matrix.')

    def __init__(self, data, kdims=None, vdims=None, extents=None, **params):
        if data is None or (isinstance(data, list) and data == []):
            data = np.zeros((0, 0))
        if extents is None:
            d1, d2 = data.shape[:2]
            extents = (0, 0, d2, d1)
        super().__init__(data, kdims=kdims, vdims=vdims, extents=extents, **params)

    def __getitem__(self, slices):
        if slices in self.dimensions():
            return self.dimension_values(slices)
        slices = util.process_ellipses(self, slices)
        if not isinstance(slices, tuple):
            slices = (slices, slice(None))
        elif len(slices) > 2 + self.depth:
            raise KeyError('Can only slice %d dimensions' % 2 + self.depth)
        elif len(slices) == 3 and slices[-1] not in [self.vdims[0].name, slice(None)]:
            raise KeyError(f'{self.vdims[0].name!r} is the only selectable value dimension')
        slc_types = [isinstance(sl, slice) for sl in slices[:2]]
        data = self.data.__getitem__(slices[:2][::-1])
        if all(slc_types):
            return self.clone(data, extents=None)
        elif not any(slc_types):
            return data
        else:
            return self.clone(np.expand_dims(data, axis=slc_types.index(True)), extents=None)

    def range(self, dim, data_range=True, dimension_range=True):
        idx = self.get_dimension_index(dim)
        if data_range and idx == 2:
            dimension = self.get_dimension(dim)
            if self.data.size == 0:
                return (np.nan, np.nan)
            lower, upper = (np.nanmin(self.data), np.nanmax(self.data))
            if not dimension_range:
                return (lower, upper)
            return util.dimension_range(lower, upper, dimension.range, dimension.soft_range)
        return super().range(dim, data_range, dimension_range)

    def dimension_values(self, dim, expanded=True, flat=True):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = self.get_dimension_index(dim)
        if not expanded and dim_idx == 0:
            return np.array(range(self.data.shape[1]))
        elif not expanded and dim_idx == 1:
            return np.array(range(self.data.shape[0]))
        elif dim_idx in [0, 1]:
            values = np.mgrid[0:self.data.shape[1], 0:self.data.shape[0]][dim_idx]
            return values.flatten() if flat else values
        elif dim_idx == 2:
            arr = self.data.T
            return arr.flatten() if flat else arr
        else:
            return super().dimension_values(dim)

    def sample(self, samples=None, bounds=None, **sample_values):
        """
        Sample the Raster along one or both of its dimensions,
        returning a reduced dimensionality type, which is either
        a ItemTable, Curve or Scatter. If two dimension samples
        and a new_xaxis is provided the sample will be the value
        of the sampled unit indexed by the value in the new_xaxis
        tuple.
        """
        if samples is None:
            samples = []
        if isinstance(samples, tuple):
            X, Y = samples
            samples = zip(X, Y)
        params = dict(self.param.values(onlychanged=True), vdims=self.vdims)
        if len(sample_values) == self.ndims or len(samples):
            if not len(samples):
                samples = zip(*[c if isinstance(c, list) else [c] for _, c in sorted([(self.get_dimension_index(k), v) for k, v in sample_values.items()])])
            table_data = [c + (self._zdata[self._coord2matrix(c)],) for c in samples]
            params['kdims'] = self.kdims
            return Table(table_data, **params)
        else:
            dimension, sample_coord = next(iter(sample_values.items()))
            if isinstance(sample_coord, slice):
                raise ValueError('Raster sampling requires coordinates not slices,use regular slicing syntax.')
            sample_ind = self.get_dimension_index(dimension)
            if sample_ind is None:
                raise Exception(f'Dimension {dimension} not found during sampling')
            other_dimension = [d for i, d in enumerate(self.kdims) if i != sample_ind]
            sample = [slice(None) for i in range(self.ndims)]
            coord_fn = (lambda v: (v, 0)) if not sample_ind else lambda v: (0, v)
            sample[sample_ind] = self._coord2matrix(coord_fn(sample_coord))[abs(sample_ind - 1)]
            x_vals = self.dimension_values(other_dimension[0].name, False)
            ydata = self._zdata[tuple(sample[::-1])]
            if hasattr(self, 'bounds') and sample_ind == 0:
                ydata = ydata[::-1]
            data = list(zip(x_vals, ydata))
            params['kdims'] = other_dimension
            return Curve(data, **params)

    def reduce(self, dimensions=None, function=None, **reduce_map):
        """
        Reduces the Raster using functions provided via the
        kwargs, where the keyword is the dimension to be reduced.
        Optionally a label_prefix can be provided to prepend to
        the result Element label.
        """
        function, dims = self._reduce_map(dimensions, function, reduce_map)
        if len(dims) == self.ndims:
            if isinstance(function, np.ufunc):
                return function.reduce(self.data, axis=None)
            else:
                return function(self.data)
        else:
            dimension = dims[0]
            other_dimension = [d for d in self.kdims if d.name != dimension]
            oidx = self.get_dimension_index(other_dimension[0])
            x_vals = self.dimension_values(other_dimension[0].name, False)
            reduced = function(self._zdata, axis=oidx)
            if oidx and hasattr(self, 'bounds'):
                reduced = reduced[::-1]
            data = zip(x_vals, reduced)
            params = dict(dict(self.param.values(onlychanged=True)), kdims=other_dimension, vdims=self.vdims)
            params.pop('bounds', None)
            params.pop('extents', None)
            return Table(data, **params)

    @property
    def depth(self):
        return len(self.vdims)

    @property
    def _zdata(self):
        return self.data

    def _coord2matrix(self, coord):
        return (int(round(coord[1])), int(round(coord[0])))

    def __len__(self):
        return np.prod(self._zdata.shape)