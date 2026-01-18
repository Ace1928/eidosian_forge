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
def _get_nth_label_width(self, nth):
    """Return the width of the *nth* label, in pixels."""
    fig = self.axes.figure
    renderer = fig._get_renderer()
    return Text(0, 0, self.get_text(self.labelLevelList[nth], self.labelFmt), figure=fig, fontproperties=self._label_font_props).get_window_extent(renderer).width