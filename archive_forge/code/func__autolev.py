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
def _autolev(self, N):
    """
        Select contour levels to span the data.

        The target number of levels, *N*, is used only when the
        scale is not log and default locator is used.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        """
    if self.locator is None:
        if self.logscale:
            self.locator = ticker.LogLocator()
        else:
            self.locator = ticker.MaxNLocator(N + 1, min_n_ticks=1)
    lev = self.locator.tick_values(self.zmin, self.zmax)
    try:
        if self.locator._symmetric:
            return lev
    except AttributeError:
        pass
    under = np.nonzero(lev < self.zmin)[0]
    i0 = under[-1] if len(under) else 0
    over = np.nonzero(lev > self.zmax)[0]
    i1 = over[0] + 1 if len(over) else len(lev)
    if self.extend in ('min', 'both'):
        i0 += 1
    if self.extend in ('max', 'both'):
        i1 -= 1
    if i1 - i0 < 3:
        i0, i1 = (0, len(lev))
    return lev[i0:i1]