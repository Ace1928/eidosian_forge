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
def _get_xy_display(self):
    """
        Get the (possibly unit converted) transformed x, y in display coords.
        """
    x, y = self.get_unitless_position()
    return self.get_transform().transform((x, y))