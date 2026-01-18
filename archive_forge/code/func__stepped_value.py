from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _stepped_value(self, val):
    """Return *val* coerced to closest number in the ``valstep`` grid."""
    if isinstance(self.valstep, Number):
        val = self.valmin + round((val - self.valmin) / self.valstep) * self.valstep
    elif self.valstep is not None:
        valstep = np.asanyarray(self.valstep)
        if valstep.ndim != 1:
            raise ValueError(f'valstep must have 1 dimension but has {valstep.ndim}')
        val = valstep[np.argmin(np.abs(valstep - val))]
    return val