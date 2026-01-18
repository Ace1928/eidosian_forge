import numpy as np
from matplotlib import _docstring
from matplotlib.contour import ContourSet
from matplotlib.tri._triangulation import Triangulation
@_docstring.dedent_interpd
class TriContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions for
    a triangular grid.

    This class is typically not instantiated directly by the user but by
    `~.Axes.tricontour` and `~.Axes.tricontourf`.

    %(contour_set_attributes)s
    """

    def __init__(self, ax, *args, **kwargs):
        """
        Draw triangular grid contour lines or filled regions,
        depending on whether keyword arg *filled* is False
        (default) or True.

        The first argument of the initializer must be an `~.axes.Axes`
        object.  The remaining arguments and keyword arguments
        are described in the docstring of `~.Axes.tricontour`.
        """
        super().__init__(ax, *args, **kwargs)

    def _process_args(self, *args, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], TriContourSet):
            C = args[0]._contour_generator
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
        else:
            from matplotlib import _tri
            tri, z = self._contour_args(args, kwargs)
            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
            self._mins = [tri.x.min(), tri.y.min()]
            self._maxs = [tri.x.max(), tri.y.max()]
        self._contour_generator = C
        return kwargs

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