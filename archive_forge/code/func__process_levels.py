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
def _process_levels(self):
    """
        Assign values to :attr:`layers` based on :attr:`levels`,
        adding extended layers as needed if contours are filled.

        For line contours, layers simply coincide with levels;
        a line is a thin layer.  No extended levels are needed
        with line contours.
        """
    self._levels = list(self.levels)
    if self.logscale:
        lower, upper = (1e-250, 1e+250)
    else:
        lower, upper = (-1e+250, 1e+250)
    if self.extend in ('both', 'min'):
        self._levels.insert(0, lower)
    if self.extend in ('both', 'max'):
        self._levels.append(upper)
    self._levels = np.asarray(self._levels)
    if not self.filled:
        self.layers = self.levels
        return
    if self.logscale:
        self.layers = np.sqrt(self._levels[:-1]) * np.sqrt(self._levels[1:])
    else:
        self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])