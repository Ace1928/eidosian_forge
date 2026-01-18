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
def _update_ticks(self, renderer=None):
    if renderer is None:
        renderer = self.figure._get_renderer()
    dpi_cor = renderer.points_to_pixels(1.0)
    if self.major_ticks.get_visible() and self.major_ticks.get_tick_out():
        ticklabel_pad = self.major_ticks._ticksize * dpi_cor
        self.major_ticklabels._external_pad = ticklabel_pad
        self.minor_ticklabels._external_pad = ticklabel_pad
    else:
        self.major_ticklabels._external_pad = 0
        self.minor_ticklabels._external_pad = 0
    majortick_iter, minortick_iter = self._axis_artist_helper.get_tick_iterators(self.axes)
    tick_loc_angle, ticklabel_loc_angle_label = self._get_tick_info(majortick_iter)
    self.major_ticks.set_locs_angles(tick_loc_angle)
    self.major_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)
    tick_loc_angle, ticklabel_loc_angle_label = self._get_tick_info(minortick_iter)
    self.minor_ticks.set_locs_angles(tick_loc_angle)
    self.minor_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)