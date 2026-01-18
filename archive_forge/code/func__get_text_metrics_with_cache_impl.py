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
@functools.lru_cache(4096)
def _get_text_metrics_with_cache_impl(renderer_ref, text, fontprop, ismath, dpi):
    return renderer_ref().get_text_width_height_descent(text, fontprop, ismath)