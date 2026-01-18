import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
class RadialLocator(mticker.Locator):
    """
    Used to locate radius ticks.

    Ensures that all ticks are strictly positive.  For all other tasks, it
    delegates to the base `.Locator` (which may be different depending on the
    scale of the *r*-axis).
    """

    def __init__(self, base, axes=None):
        self.base = base
        self._axes = axes

    def set_axis(self, axis):
        self.base.set_axis(axis)

    def __call__(self):
        if self._axes:
            if _is_full_circle_rad(*self._axes.viewLim.intervalx):
                rorigin = self._axes.get_rorigin() * self._axes.get_rsign()
                if self._axes.get_rmin() <= rorigin:
                    return [tick for tick in self.base() if tick > rorigin]
        return self.base()

    def _zero_in_bounds(self):
        """
        Return True if zero is within the valid values for the
        scale of the radial axis.
        """
        vmin, vmax = self._axes.yaxis._scale.limit_range_for_scale(0, 1, 1e-05)
        return vmin == 0

    def nonsingular(self, vmin, vmax):
        if self._zero_in_bounds() and (vmin, vmax) == (-np.inf, np.inf):
            return (0, 1)
        else:
            return self.base.nonsingular(vmin, vmax)

    def view_limits(self, vmin, vmax):
        vmin, vmax = self.base.view_limits(vmin, vmax)
        if self._zero_in_bounds() and vmax > vmin:
            vmin = min(0, vmin)
        return mtransforms.nonsingular(vmin, vmax)