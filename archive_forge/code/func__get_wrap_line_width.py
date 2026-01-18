import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def _get_wrap_line_width(self):
    """
        Return the maximum line width for wrapping text based on the current
        orientation.
        """
    x0, y0 = self.get_transform().transform(self.get_position())
    figure_box = self.get_figure().get_window_extent()
    alignment = self.get_horizontalalignment()
    self.set_rotation_mode('anchor')
    rotation = self.get_rotation()
    left = self._get_dist_to_box(rotation, x0, y0, figure_box)
    right = self._get_dist_to_box((180 + rotation) % 360, x0, y0, figure_box)
    if alignment == 'left':
        line_width = left
    elif alignment == 'right':
        line_width = right
    else:
        line_width = 2 * min(left, right)
    return line_width