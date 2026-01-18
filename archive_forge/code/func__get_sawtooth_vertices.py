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
def _get_sawtooth_vertices(self, x0, y0, width, height, mutation_size):
    pad = mutation_size * self.pad
    if self.tooth_size is None:
        tooth_size = self.pad * 0.5 * mutation_size
    else:
        tooth_size = self.tooth_size * mutation_size
    hsz = tooth_size / 2
    width = width + 2 * pad - tooth_size
    height = height + 2 * pad - tooth_size
    dsx_n = round((width - tooth_size) / (tooth_size * 2)) * 2
    dsy_n = round((height - tooth_size) / (tooth_size * 2)) * 2
    x0, y0 = (x0 - pad + hsz, y0 - pad + hsz)
    x1, y1 = (x0 + width, y0 + height)
    xs = [x0, *np.linspace(x0 + hsz, x1 - hsz, 2 * dsx_n + 1), *([x1, x1 + hsz, x1, x1 - hsz] * dsy_n)[:2 * dsy_n + 2], x1, *np.linspace(x1 - hsz, x0 + hsz, 2 * dsx_n + 1), *([x0, x0 - hsz, x0, x0 + hsz] * dsy_n)[:2 * dsy_n + 2]]
    ys = [*([y0, y0 - hsz, y0, y0 + hsz] * dsx_n)[:2 * dsx_n + 2], y0, *np.linspace(y0 + hsz, y1 - hsz, 2 * dsy_n + 1), *([y1, y1 + hsz, y1, y1 - hsz] * dsx_n)[:2 * dsx_n + 2], y1, *np.linspace(y1 - hsz, y0 + hsz, 2 * dsy_n + 1)]
    return [*zip(xs, ys), (xs[0], ys[0])]