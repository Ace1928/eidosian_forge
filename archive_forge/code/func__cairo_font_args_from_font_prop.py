import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def _cairo_font_args_from_font_prop(prop):
    """
    Convert a `.FontProperties` or a `.FontEntry` to arguments that can be
    passed to `.Context.select_font_face`.
    """

    def attr(field):
        try:
            return getattr(prop, f'get_{field}')()
        except AttributeError:
            return getattr(prop, field)
    name = attr('name')
    slant = getattr(cairo, f'FONT_SLANT_{attr('style').upper()}')
    weight = attr('weight')
    weight = cairo.FONT_WEIGHT_NORMAL if font_manager.weight_dict.get(weight, weight) < 550 else cairo.FONT_WEIGHT_BOLD
    return (name, slant, weight)