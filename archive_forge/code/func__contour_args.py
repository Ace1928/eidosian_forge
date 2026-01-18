import numpy as np
from matplotlib import _docstring
from matplotlib.contour import ContourSet
from matplotlib.tri._triangulation import Triangulation
def _contour_args(self, args, kwargs):
    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    z, *args = args
    z = np.ma.asarray(z)
    if z.shape != tri.x.shape:
        raise ValueError('z array must have same length as triangulation x and y arrays')
    z_check = z[np.unique(tri.get_masked_triangles())]
    if np.ma.is_masked(z_check):
        raise ValueError('z must not contain masked points within the triangulation')
    if not np.isfinite(z_check).all():
        raise ValueError('z array must not contain non-finite values within the triangulation')
    z = np.ma.masked_invalid(z, copy=False)
    self.zmax = float(z_check.max())
    self.zmin = float(z_check.min())
    if self.logscale and self.zmin <= 0:
        func = 'contourf' if self.filled else 'contour'
        raise ValueError(f'Cannot {func} log of negative values.')
    self._process_contour_level_args(args, z.dtype)
    return (tri, z)