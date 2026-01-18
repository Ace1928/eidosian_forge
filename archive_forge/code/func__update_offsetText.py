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
def _update_offsetText(self):
    self.offsetText.set_text(self.axis.major.formatter.get_offset())
    self.offsetText.set_size(self.major_ticklabels.get_size())
    offset = self.major_ticklabels.get_pad() + self.major_ticklabels.get_size() + 2
    self.offsetText.xyann = (0, offset)