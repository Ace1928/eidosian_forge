import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class MultipleLocator(Locator):
    """
    Set a tick on each integer multiple of the *base* plus an *offset* within
    the view interval.
    """

    def __init__(self, base=1.0, offset=0.0):
        """
        Parameters
        ----------
        base : float > 0
            Interval between ticks.
        offset : float
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
        self._edge = _Edge_integer(base, 0)
        self._offset = offset

    def set_params(self, base=None, offset=None):
        """
        Set parameters within this locator.

        Parameters
        ----------
        base : float > 0
            Interval between ticks.
        offset : float
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
        if base is not None:
            self._edge = _Edge_integer(base, 0)
        if offset is not None:
            self._offset = offset

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = (vmax, vmin)
        step = self._edge.step
        vmin -= self._offset
        vmax -= self._offset
        vmin = self._edge.ge(vmin) * step
        n = (vmax - vmin + 0.001 * step) // step
        locs = vmin - step + np.arange(n + 3) * step + self._offset
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest tick values that contain the data.
        """
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = self._edge.le(dmin - self._offset) * self._edge.step + self._offset
            vmax = self._edge.ge(dmax - self._offset) * self._edge.step + self._offset
            if vmin == vmax:
                vmin -= 1
                vmax += 1
        else:
            vmin = dmin
            vmax = dmax
        return mtransforms.nonsingular(vmin, vmax)