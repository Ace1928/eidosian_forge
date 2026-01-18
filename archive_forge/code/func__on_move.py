from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def _on_move(self, event):
    """
        Mouse moving.

        By default, button-1 rotates, button-2 pans, and button-3 zooms;
        these buttons can be modified via `mouse_init`.
        """
    if not self.button_pressed:
        return
    if self.get_navigate_mode() is not None:
        return
    if self.M is None:
        return
    x, y = (event.xdata, event.ydata)
    if x is None or event.inaxes != self:
        return
    dx, dy = (x - self._sx, y - self._sy)
    w = self._pseudo_w
    h = self._pseudo_h
    if self.button_pressed in self._rotate_btn:
        if dx == 0 and dy == 0:
            return
        roll = np.deg2rad(self.roll)
        delev = -(dy / h) * 180 * np.cos(roll) + dx / w * 180 * np.sin(roll)
        dazim = -(dy / h) * 180 * np.sin(roll) - dx / w * 180 * np.cos(roll)
        elev = self.elev + delev
        azim = self.azim + dazim
        self.view_init(elev=elev, azim=azim, roll=roll, share=True)
        self.stale = True
    elif self.button_pressed in self._pan_btn:
        px, py = self.transData.transform([self._sx, self._sy])
        self.start_pan(px, py, 2)
        self.drag_pan(2, None, event.x, event.y)
        self.end_pan()
    elif self.button_pressed in self._zoom_btn:
        scale = h / (h - dy)
        self._scale_axis_limits(scale, scale, scale)
    self._sx, self._sy = (x, y)
    self.figure.canvas.draw_idle()