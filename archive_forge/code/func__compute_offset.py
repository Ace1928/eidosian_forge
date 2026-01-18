import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _compute_offset(self):
    locs = self.locs
    vmin, vmax = sorted(self.axis.get_view_interval())
    locs = np.asarray(locs)
    locs = locs[(vmin <= locs) & (locs <= vmax)]
    if not len(locs):
        self.offset = 0
        return
    lmin, lmax = (locs.min(), locs.max())
    if lmin == lmax or lmin <= 0 <= lmax:
        self.offset = 0
        return
    abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
    sign = math.copysign(1, lmin)
    oom_max = np.ceil(math.log10(abs_max))
    oom = 1 + next((oom for oom in itertools.count(oom_max, -1) if abs_min // 10 ** oom != abs_max // 10 ** oom))
    if (abs_max - abs_min) / 10 ** oom <= 0.01:
        oom = 1 + next((oom for oom in itertools.count(oom_max, -1) if abs_max // 10 ** oom - abs_min // 10 ** oom > 1))
    n = self._offset_threshold - 1
    self.offset = sign * (abs_max // 10 ** oom) * 10 ** oom if abs_max // 10 ** oom >= 10 ** n else 0