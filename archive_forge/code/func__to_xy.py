from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def _to_xy(self, values, const):
    """
        Create a (*values.shape, 2)-shape array representing (x, y) pairs.

        The other coordinate is filled with the constant *const*.

        Example::

            >>> self.nth_coord = 0
            >>> self._to_xy([1, 2, 3], const=0)
            array([[1, 0],
                   [2, 0],
                   [3, 0]])
        """
    if self.nth_coord == 0:
        return np.stack(np.broadcast_arrays(values, const), axis=-1)
    elif self.nth_coord == 1:
        return np.stack(np.broadcast_arrays(const, values), axis=-1)
    else:
        raise ValueError('Unexpected nth_coord')