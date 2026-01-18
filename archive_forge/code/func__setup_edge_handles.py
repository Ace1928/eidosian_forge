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
def _setup_edge_handles(self, props):
    if self.direction == 'horizontal':
        positions = self.ax.get_xbound()
    else:
        positions = self.ax.get_ybound()
    self._edge_handles = ToolLineHandles(self.ax, positions, direction=self.direction, line_props=props, useblit=self.useblit)