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
def _get_rendered_text_width(self, text):
    """
        Return the width of a given text string, in pixels.
        """
    w, h, d = self._renderer.get_text_width_height_descent(text, self.get_fontproperties(), cbook.is_math_text(text))
    return math.ceil(w)