import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
@_api.deprecated('3.8')
class AxesLocator:
    """
    A callable object which returns the position and size of a given
    `.AxesDivider` cell.
    """

    def __init__(self, axes_divider, nx, ny, nx1=None, ny1=None):
        """
        Parameters
        ----------
        axes_divider : `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise, location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        ny, ny1 : int
            Same as *nx* and *nx1*, but for row positions.
        """
        self._axes_divider = axes_divider
        _xrefindex = axes_divider._xrefindex
        _yrefindex = axes_divider._yrefindex
        self._nx, self._ny = (nx - _xrefindex, ny - _yrefindex)
        if nx1 is None:
            nx1 = len(self._axes_divider)
        if ny1 is None:
            ny1 = len(self._axes_divider[0])
        self._nx1 = nx1 - _xrefindex
        self._ny1 = ny1 - _yrefindex

    def __call__(self, axes, renderer):
        _xrefindex = self._axes_divider._xrefindex
        _yrefindex = self._axes_divider._yrefindex
        return self._axes_divider.locate(self._nx + _xrefindex, self._ny + _yrefindex, self._nx1 + _xrefindex, self._ny1 + _yrefindex, axes, renderer)

    def get_subplotspec(self):
        return self._axes_divider.get_subplotspec()