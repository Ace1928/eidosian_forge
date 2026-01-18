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
def _get_tick_info(self, tick_iter):
    """
        Return a pair of:

        - list of locs and angles for ticks
        - list of locs, angles and labels for ticklabels.
        """
    ticks_loc_angle = []
    ticklabels_loc_angle_label = []
    ticklabel_add_angle = self._ticklabel_add_angle
    for loc, angle_normal, angle_tangent, label in tick_iter:
        angle_label = angle_tangent - 90 + ticklabel_add_angle
        angle_tick = angle_normal if 90 <= (angle_label - angle_normal) % 360 <= 270 else angle_normal + 180
        ticks_loc_angle.append([loc, angle_tick])
        ticklabels_loc_angle_label.append([loc, angle_label, label])
    return (ticks_loc_angle, ticklabels_loc_angle_label)