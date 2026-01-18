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