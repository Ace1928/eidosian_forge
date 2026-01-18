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
def _get_data_coords(self, event):
    """Return *event*'s data coordinates in this widget's Axes."""
    return (event.xdata, event.ydata) if event.inaxes is self.ax else self.ax.transData.inverted().transform((event.x, event.y))