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
def _get_path_in_displaycoord(self):
    """Return the mutated path of the arrow in display coordinates."""
    dpi_cor = self._dpi_cor
    posA = self._get_xy(self.xy1, self.coords1, self.axesA)
    posB = self._get_xy(self.xy2, self.coords2, self.axesB)
    path = self.get_connectionstyle()(posA, posB, patchA=self.patchA, patchB=self.patchB, shrinkA=self.shrinkA * dpi_cor, shrinkB=self.shrinkB * dpi_cor)
    path, fillable = self.get_arrowstyle()(path, self.get_mutation_scale() * dpi_cor, self.get_linewidth() * dpi_cor, self.get_mutation_aspect())
    return (path, fillable)