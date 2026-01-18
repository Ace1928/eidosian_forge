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
def _set_view_from_bbox(self, bbox, direction='in', mode=None, twinx=False, twiny=False):
    """
        Zoom in or out of the bounding box.

        Will center the view in the center of the bounding box, and zoom by
        the ratio of the size of the bounding box to the size of the Axes3D.
        """
    start_x, start_y, stop_x, stop_y = bbox
    if mode == 'x':
        start_y = self.bbox.min[1]
        stop_y = self.bbox.max[1]
    elif mode == 'y':
        start_x = self.bbox.min[0]
        stop_x = self.bbox.max[0]
    start_x, stop_x = np.clip(sorted([start_x, stop_x]), self.bbox.min[0], self.bbox.max[0])
    start_y, stop_y = np.clip(sorted([start_y, stop_y]), self.bbox.min[1], self.bbox.max[1])
    zoom_center_x = (start_x + stop_x) / 2
    zoom_center_y = (start_y + stop_y) / 2
    ax_center_x = (self.bbox.max[0] + self.bbox.min[0]) / 2
    ax_center_y = (self.bbox.max[1] + self.bbox.min[1]) / 2
    self.start_pan(zoom_center_x, zoom_center_y, 2)
    self.drag_pan(2, None, ax_center_x, ax_center_y)
    self.end_pan()
    dx = abs(start_x - stop_x)
    dy = abs(start_y - stop_y)
    scale_u = dx / (self.bbox.max[0] - self.bbox.min[0])
    scale_v = dy / (self.bbox.max[1] - self.bbox.min[1])
    scale = max(scale_u, scale_v)
    if direction == 'out':
        scale = 1 / scale
    self._zoom_data_limits(scale, scale, scale)