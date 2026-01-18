import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
@_register_style(_style_list)
class DArrow:
    """A box in the shape of a two-way arrow."""

    def __init__(self, pad=0.3):
        """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
        self.pad = pad

    def __call__(self, x0, y0, width, height, mutation_size):
        pad = mutation_size * self.pad
        height = height + 2 * pad
        x0, y0 = (x0 - pad, y0 - pad)
        x1, y1 = (x0 + width, y0 + height)
        dx = (y1 - y0) / 2
        dxx = dx / 2
        x0 = x0 + pad / 1.4
        return Path._create_closed([(x0 + dxx, y0), (x1, y0), (x1, y0 - dxx), (x1 + dx + dxx, y0 + dx), (x1, y1 + dxx), (x1, y1), (x0 + dxx, y1), (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx), (x0 + dxx, y0 - dxx), (x0 + dxx, y0)])