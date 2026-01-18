from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
class _PointLike(Glyph):
    """Shared methods between Point and Line"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.x, self.y)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y must be real')

    @property
    def x_label(self):
        return self.x

    @property
    def y_label(self):
        return self.y

    def required_columns(self):
        return [self.x, self.y]

    def compute_x_bounds(self, df):
        bounds = self._compute_bounds(df[self.x])
        return self.maybe_expand_bounds(bounds)

    def compute_y_bounds(self, df):
        bounds = self._compute_bounds(df[self.y])
        return self.maybe_expand_bounds(bounds)

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin(df[self.x].values).item(), np.nanmax(df[self.x].values).item(), np.nanmin(df[self.y].values).item(), np.nanmax(df[self.y].values).item()]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y_extents = (np.nanmin(r[:, 2]), np.nanmax(r[:, 3]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))