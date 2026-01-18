from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
class _GeometryLike(Glyph):

    def __init__(self, geometry):
        self.geometry = geometry
        self._cached_bounds = None

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.geometry,)

    @property
    def geom_dtypes(self):
        if spatialpandas:
            from spatialpandas.geometry import GeometryDtype
            return (GeometryDtype,)
        else:
            return ()

    def validate(self, in_dshape):
        if not isinstance(in_dshape[str(self.geometry)], self.geom_dtypes):
            raise ValueError('{col} must be an array with one of the following types: {typs}'.format(col=self.geometry, typs=', '.join((typ.__name__ for typ in self.geom_dtypes))))

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def required_columns(self):
        return [self.geometry]

    def compute_x_bounds(self, df):
        col = df[self.geometry]
        if isinstance(col.dtype, gpd_GeometryDtype):
            if self._cached_bounds is None:
                self._cached_bounds = col.total_bounds
            bounds = self._cached_bounds[::2]
        else:
            bounds = col.array.total_bounds_x
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        col = df[self.geometry]
        if isinstance(col.dtype, gpd_GeometryDtype):
            if self._cached_bounds is None:
                self._cached_bounds = col.total_bounds
            bounds = self._cached_bounds[1::2]
        else:
            bounds = col.array.total_bounds_y
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        total_bounds = ddf[self.geometry].total_bounds
        x_extents = (total_bounds[0], total_bounds[2])
        y_extents = (total_bounds[1], total_bounds[3])
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))