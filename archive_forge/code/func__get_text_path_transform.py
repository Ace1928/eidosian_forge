from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def _get_text_path_transform(self, x, y, s, prop, angle, ismath):
    """
        Return the text path and transform.

        Parameters
        ----------
        x : float
            The x location of the text in display coords.
        y : float
            The y location of the text baseline in display coords.
        s : str
            The text to be converted.
        prop : `~matplotlib.font_manager.FontProperties`
            The font property.
        angle : float
            Angle in degrees to render the text at.
        ismath : bool or "TeX"
            If True, use mathtext parser. If "TeX", use tex for rendering.
        """
    text2path = self._text2path
    fontsize = self.points_to_pixels(prop.get_size_in_points())
    verts, codes = text2path.get_text_path(prop, s, ismath=ismath)
    path = Path(verts, codes)
    angle = np.deg2rad(angle)
    if self.flipy():
        width, height = self.get_canvas_width_height()
        transform = Affine2D().scale(fontsize / text2path.FONT_SCALE).rotate(angle).translate(x, height - y)
    else:
        transform = Affine2D().scale(fontsize / text2path.FONT_SCALE).rotate(angle).translate(x, y)
    return (path, transform)