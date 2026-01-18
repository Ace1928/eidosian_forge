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
def _add_box(self):
    self._box = RectangleSelector(self.ax, onselect=lambda *args, **kwargs: None, useblit=self.useblit, grab_range=self.grab_range, handle_props=self._box_handle_props, props=self._box_props, interactive=True)
    self._box._state_modifier_keys.pop('rotate')
    self._box.connect_event('motion_notify_event', self._scale_polygon)
    self._update_box()
    self._box._allow_creation = False
    self._box._selection_completed = True
    self._draw_polygon()