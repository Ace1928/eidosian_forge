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
def _process_colors(self):
    """
        Color argument processing for contouring.

        Note that we base the colormapping on the contour levels
        and layers, not on the actual range of the Z values.  This
        means we don't have to worry about bad values in Z, and we
        always have the full dynamic range available for the selected
        levels.

        The color is based on the midpoint of the layer, except for
        extended end layers.  By default, the norm vmin and vmax
        are the extreme values of the non-extended levels.  Hence,
        the layer color extremes are not the extreme values of
        the colormap itself, but approach those values as the number
        of levels increases.  An advantage of this scheme is that
        line contours, when added to filled contours, take on
        colors that are consistent with those of the filled regions;
        for example, a contour line on the boundary between two
        regions will have a color intermediate between those
        of the regions.

        """
    self.monochrome = self.cmap.monochrome
    if self.colors is not None:
        i0, i1 = (0, len(self.levels))
        if self.filled:
            i1 -= 1
            if self.extend in ('both', 'min'):
                i0 -= 1
            if self.extend in ('both', 'max'):
                i1 += 1
        self.cvalues = list(range(i0, i1))
        self.set_norm(mcolors.NoNorm())
    else:
        self.cvalues = self.layers
    self.norm.autoscale_None(self.levels)
    self.set_array(self.cvalues)
    self.update_scalarmappable()
    if self.extend in ('both', 'max', 'min'):
        self.norm.clip = False