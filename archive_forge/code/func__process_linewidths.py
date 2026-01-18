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
def _process_linewidths(self, linewidths):
    Nlev = len(self.levels)
    if linewidths is None:
        default_linewidth = mpl.rcParams['contour.linewidth']
        if default_linewidth is None:
            default_linewidth = mpl.rcParams['lines.linewidth']
        return [default_linewidth] * Nlev
    elif not np.iterable(linewidths):
        return [linewidths] * Nlev
    else:
        linewidths = list(linewidths)
        return (linewidths * math.ceil(Nlev / len(linewidths)))[:Nlev]