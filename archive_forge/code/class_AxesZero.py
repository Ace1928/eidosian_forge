from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
class AxesZero(Axes):

    def clear(self):
        super().clear()
        new_floating_axis = self.get_grid_helper().new_floating_axis
        self._axislines.update(xzero=new_floating_axis(nth_coord=0, value=0.0, axis_direction='bottom', axes=self), yzero=new_floating_axis(nth_coord=1, value=0.0, axis_direction='left', axes=self))
        for k in ['xzero', 'yzero']:
            self._axislines[k].line.set_clip_path(self.patch)
            self._axislines[k].set_visible(False)