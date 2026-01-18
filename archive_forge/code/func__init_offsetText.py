from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def _init_offsetText(self, direction):
    x, y, va, ha = self._offsetText_pos[direction]
    self.offsetText = mtext.Annotation('', xy=(x, y), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points', color=mpl.rcParams['xtick.color'], horizontalalignment=ha, verticalalignment=va)
    self.offsetText.set_transform(IdentityTransform())
    self.axes._set_artist_props(self.offsetText)