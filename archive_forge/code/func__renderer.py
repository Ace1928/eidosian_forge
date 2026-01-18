import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
@property
def _renderer(self):
    if not hasattr(self, '_cached_renderer'):
        self._cached_renderer = RendererCairo(self.figure.dpi)
    return self._cached_renderer