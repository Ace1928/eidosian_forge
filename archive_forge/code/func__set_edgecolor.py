import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _set_edgecolor(self, c):
    set_hatch_color = True
    if c is None:
        if mpl.rcParams['patch.force_edgecolor'] or self._edge_default or cbook._str_equal(self._original_facecolor, 'none'):
            c = self._get_default_edgecolor()
        else:
            c = 'none'
            set_hatch_color = False
    if cbook._str_lower_equal(c, 'face'):
        self._edgecolors = 'face'
        self.stale = True
        return
    self._edgecolors = mcolors.to_rgba_array(c, self._alpha)
    if set_hatch_color and len(self._edgecolors):
        self._hatch_color = tuple(self._edgecolors[0])
    self.stale = True