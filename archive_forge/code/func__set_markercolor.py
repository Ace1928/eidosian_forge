import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def _set_markercolor(self, name, has_rcdefault, val):
    if val is None:
        val = mpl.rcParams[f'lines.{name}'] if has_rcdefault else 'auto'
    attr = f'_{name}'
    current = getattr(self, attr)
    if current is None:
        self.stale = True
    else:
        neq = current != val
        if neq.any() if isinstance(neq, np.ndarray) else neq:
            self.stale = True
    setattr(self, attr, val)