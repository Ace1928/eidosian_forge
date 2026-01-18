import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
class HBoxDivider(SubplotDivider):
    """
    A `.SubplotDivider` for laying out axes horizontally, while ensuring that
    they have equal heights.

    Examples
    --------
    .. plot:: gallery/axes_grid1/demo_axes_hbox_divider.py
    """

    def new_locator(self, nx, nx1=None):
        """
        Create an axes locator callable for the specified cell.

        Parameters
        ----------
        nx, nx1 : int
            Integers specifying the column-position of the
            cell. When *nx1* is None, a single *nx*-th column is
            specified. Otherwise, location of columns spanning between *nx*
            to *nx1* (but excluding *nx1*-th column) is specified.
        """
        return super().new_locator(nx, 0, nx1, 0)

    def _locate(self, nx, ny, nx1, ny1, axes, renderer):
        nx += self._xrefindex
        nx1 += self._xrefindex
        fig_w, fig_h = self._fig.bbox.size / self._fig.dpi
        x, y, w, h = self.get_position_runtime(axes, renderer)
        summed_ws = self.get_horizontal_sizes(renderer)
        equal_hs = self.get_vertical_sizes(renderer)
        x0, y0, ox, hh = _locate(x, y, w, h, summed_ws, equal_hs, fig_w, fig_h, self.get_anchor())
        if nx1 is None:
            nx1 = -1
        x1, w1 = (x0 + ox[nx] / fig_w, (ox[nx1] - ox[nx]) / fig_w)
        y1, h1 = (y0, hh)
        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)