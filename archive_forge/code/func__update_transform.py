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
def _update_transform(self, renderer):
    ox = renderer.points_to_pixels(self._ox)
    oy = renderer.points_to_pixels(self._oy)
    self._shadow_transform.clear().translate(ox, oy)