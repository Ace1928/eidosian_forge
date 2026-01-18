from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
class ToolHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str, default: 'o'
        Shape of marker used to display handle. See `~.pyplot.plot`.
    marker_props : dict, optional
        Additional marker properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """

    @_api.make_keyword_only('3.7', 'marker')
    def __init__(self, ax, x, y, marker='o', marker_props=None, useblit=True):
        self.ax = ax
        props = {'marker': marker, 'markersize': 7, 'markerfacecolor': 'w', 'linestyle': 'none', 'alpha': 0.5, 'visible': False, 'label': '_nolegend_', **cbook.normalize_kwargs(marker_props, Line2D._alias_map)}
        self._markers = Line2D(x, y, animated=useblit, **props)
        self.ax.add_line(self._markers)

    @property
    def x(self):
        return self._markers.get_xdata()

    @property
    def y(self):
        return self._markers.get_ydata()

    @property
    def artists(self):
        return (self._markers,)

    def set_data(self, pts, y=None):
        """Set x and y positions of handles."""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        self._markers.set_data(pts)

    def set_visible(self, val):
        self._markers.set_visible(val)

    def set_animated(self, val):
        self._markers.set_animated(val)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        pts = np.column_stack([self.x, self.y])
        pts = self.ax.transData.transform(pts)
        diff = pts - [x, y]
        dist = np.hypot(*diff.T)
        min_index = np.argmin(dist)
        return (min_index, dist[min_index])