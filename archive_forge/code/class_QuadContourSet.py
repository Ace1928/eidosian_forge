from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
@_docstring.dedent_interpd
class QuadContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions.

    This class is typically not instantiated directly by the user but by
    `~.Axes.contour` and `~.Axes.contourf`.

    %(contour_set_attributes)s
    """

    def _process_args(self, *args, corner_mask=None, algorithm=None, **kwargs):
        """
        Process args and kwargs.
        """
        if args and isinstance(args[0], QuadContourSet):
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._corner_mask = args[0]._corner_mask
            contour_generator = args[0]._contour_generator
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
            self._algorithm = args[0]._algorithm
        else:
            import contourpy
            if algorithm is None:
                algorithm = mpl.rcParams['contour.algorithm']
            mpl.rcParams.validate['contour.algorithm'](algorithm)
            self._algorithm = algorithm
            if corner_mask is None:
                if self._algorithm == 'mpl2005':
                    corner_mask = False
                else:
                    corner_mask = mpl.rcParams['contour.corner_mask']
            self._corner_mask = corner_mask
            x, y, z = self._contour_args(args, kwargs)
            contour_generator = contourpy.contour_generator(x, y, z, name=self._algorithm, corner_mask=self._corner_mask, line_type=contourpy.LineType.SeparateCode, fill_type=contourpy.FillType.OuterCode, chunk_size=self.nchunk)
            t = self.get_transform()
            if t != self.axes.transData and any(t.contains_branch_seperately(self.axes.transData)):
                trans_to_data = t - self.axes.transData
                pts = np.vstack([x.flat, y.flat]).T
                transformed_pts = trans_to_data.transform(pts)
                x = transformed_pts[..., 0]
                y = transformed_pts[..., 1]
            self._mins = [ma.min(x), ma.min(y)]
            self._maxs = [ma.max(x), ma.max(y)]
        self._contour_generator = contour_generator
        return kwargs

    def _contour_args(self, args, kwargs):
        if self.filled:
            fn = 'contourf'
        else:
            fn = 'contour'
        nargs = len(args)
        if 0 < nargs <= 2:
            z, *args = args
            z = ma.asarray(z)
            x, y = self._initialize_x_y(z)
        elif 2 < nargs <= 4:
            x, y, z_orig, *args = args
            x, y, z = self._check_xyz(x, y, z_orig, kwargs)
        else:
            raise _api.nargs_error(fn, takes='from 1 to 4', given=nargs)
        z = ma.masked_invalid(z, copy=False)
        self.zmax = z.max().astype(float)
        self.zmin = z.min().astype(float)
        if self.logscale and self.zmin <= 0:
            z = ma.masked_where(z <= 0, z)
            _api.warn_external('Log scale: values of z <= 0 have been masked')
            self.zmin = z.min().astype(float)
        self._process_contour_level_args(args, z.dtype)
        return (x, y, z)

    def _check_xyz(self, x, y, z, kwargs):
        """
        Check that the shapes of the input arrays match; if x and y are 1D,
        convert them to 2D using meshgrid.
        """
        x, y = self.axes._process_unit_info([('x', x), ('y', y)], kwargs)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = ma.asarray(z)
        if z.ndim != 2:
            raise TypeError(f'Input z must be 2D, not {z.ndim}D')
        if z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f'Input z must be at least a (2, 2) shaped array, but has shape {z.shape}')
        Ny, Nx = z.shape
        if x.ndim != y.ndim:
            raise TypeError(f'Number of dimensions of x ({x.ndim}) and y ({y.ndim}) do not match')
        if x.ndim == 1:
            nx, = x.shape
            ny, = y.shape
            if nx != Nx:
                raise TypeError(f'Length of x ({nx}) must match number of columns in z ({Nx})')
            if ny != Ny:
                raise TypeError(f'Length of y ({ny}) must match number of rows in z ({Ny})')
            x, y = np.meshgrid(x, y)
        elif x.ndim == 2:
            if x.shape != z.shape:
                raise TypeError(f'Shapes of x {x.shape} and z {z.shape} do not match')
            if y.shape != z.shape:
                raise TypeError(f'Shapes of y {y.shape} and z {z.shape} do not match')
        else:
            raise TypeError(f'Inputs x and y must be 1D or 2D, not {x.ndim}D')
        return (x, y, z)

    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i, j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        if z.ndim != 2:
            raise TypeError(f'Input z must be 2D, not {z.ndim}D')
        elif z.shape[0] < 2 or z.shape[1] < 2:
            raise TypeError(f'Input z must be at least a (2, 2) shaped array, but has shape {z.shape}')
        else:
            Ny, Nx = z.shape
        if self.origin is None:
            if self.extent is None:
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                x0, x1, y0, y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)
        if self.extent is None:
            x0, x1, y0, y1 = (0, Nx, 0, Ny)
        else:
            x0, x1, y0, y1 = self.extent
        dx = (x1 - x0) / Nx
        dy = (y1 - y0) / Ny
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return np.meshgrid(x, y)