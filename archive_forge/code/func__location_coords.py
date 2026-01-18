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
def _location_coords(self, xv, yv, renderer):
    """
        Return the location on the axis pane underneath the cursor as a string.
        """
    p1, pane_idx = self._calc_coord(xv, yv, renderer)
    xs = self.format_xdata(p1[0])
    ys = self.format_ydata(p1[1])
    zs = self.format_zdata(p1[2])
    if pane_idx == 0:
        coords = f'x pane={xs}, y={ys}, z={zs}'
    elif pane_idx == 1:
        coords = f'x={xs}, y pane={ys}, z={zs}'
    elif pane_idx == 2:
        coords = f'x={xs}, y={ys}, z pane={zs}'
    return coords