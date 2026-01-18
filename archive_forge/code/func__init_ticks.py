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
def _init_ticks(self, **kwargs):
    axis_name = self.axis.axis_name
    trans = self._axis_artist_helper.get_tick_transform(self.axes) + self.offset_transform
    self.major_ticks = Ticks(kwargs.get('major_tick_size', mpl.rcParams[f'{axis_name}tick.major.size']), axis=self.axis, transform=trans)
    self.minor_ticks = Ticks(kwargs.get('minor_tick_size', mpl.rcParams[f'{axis_name}tick.minor.size']), axis=self.axis, transform=trans)
    size = mpl.rcParams[f'{axis_name}tick.labelsize']
    self.major_ticklabels = TickLabels(axis=self.axis, axis_direction=self._axis_direction, figure=self.axes.figure, transform=trans, fontsize=size, pad=kwargs.get('major_tick_pad', mpl.rcParams[f'{axis_name}tick.major.pad']))
    self.minor_ticklabels = TickLabels(axis=self.axis, axis_direction=self._axis_direction, figure=self.axes.figure, transform=trans, fontsize=size, pad=kwargs.get('minor_tick_pad', mpl.rcParams[f'{axis_name}tick.minor.pad']))