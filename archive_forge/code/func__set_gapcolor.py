import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _set_gapcolor(self, gapcolor):
    if gapcolor is not None:
        gapcolor = mcolors.to_rgba_array(gapcolor, self._alpha)
    self._gapcolor = gapcolor
    self.stale = True