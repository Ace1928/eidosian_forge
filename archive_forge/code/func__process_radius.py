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
def _process_radius(self, radius):
    if radius is not None:
        return radius
    if isinstance(self._picker, Number):
        _radius = self._picker
    elif self.get_edgecolor()[3] == 0:
        _radius = 0
    else:
        _radius = self.get_linewidth()
    return _radius