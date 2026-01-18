import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
class VBoxDivider(SubplotDivider):
    """
    A `.SubplotDivider` for laying out axes vertically, while ensuring that
    they have equal widths.
    """

    def new_locator(self, ny, ny1=None):
        """
        Create an axes locator callable for the specified cell.

        Parameters
        ----------
        ny, ny1 : int
            Integers specifying the row-position of the
            cell. When *ny1* is None, a single *ny*-th row is
            specified. Otherwise, location of rows spanning between *ny*
            to *ny1* (but excluding *ny1*-th row) is specified.
        """
        return super().new_locator(0, ny, 0, ny1)

    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        ny += self._yrefindex
        ny1 += self._yrefindex
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_hs = self.get_vertical_sizes(renderer)
        equal_ws = self.get_horizontal_sizes(renderer)
        y0, x0, oy, ww = _locate(y, x, h, w, summed_hs, equal_ws, fig_h, fig_w, self.get_anchor())
        if ny1 is None:
            ny1 = -1
        x1, w1 = (x0, ww)
        y1, h1 = (y0 + oy[ny] / fig_h, (oy[ny1] - oy[ny]) / fig_h)
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)