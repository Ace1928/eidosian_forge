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