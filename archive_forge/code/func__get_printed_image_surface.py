import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def _get_printed_image_surface(self):
    self._renderer.dpi = self.figure.dpi
    width, height = self.get_width_height()
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    self._renderer.set_context(cairo.Context(surface))
    self.figure.draw(self._renderer)
    return surface