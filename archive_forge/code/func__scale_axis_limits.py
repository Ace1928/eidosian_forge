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
def _scale_axis_limits(self, scale_x, scale_y, scale_z):
    """
        Keeping the center of the x, y, and z data axes fixed, scale their
        limits by scale factors. A scale factor > 1 zooms out and a scale
        factor < 1 zooms in.

        Parameters
        ----------
        scale_x : float
            Scale factor for the x data axis.
        scale_y : float
            Scale factor for the y data axis.
        scale_z : float
            Scale factor for the z data axis.
        """
    cx, cy, cz, dx, dy, dz = self._get_w_centers_ranges()
    self.set_xlim3d(cx - dx * scale_x / 2, cx + dx * scale_x / 2)
    self.set_ylim3d(cy - dy * scale_y / 2, cy + dy * scale_y / 2)
    self.set_zlim3d(cz - dz * scale_z / 2, cz + dz * scale_z / 2)