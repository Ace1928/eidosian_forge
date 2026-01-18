import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _set_order_of_magnitude(self):
    if not self._scientific:
        self.orderOfMagnitude = 0
        return
    if self._powerlimits[0] == self._powerlimits[1] != 0:
        self.orderOfMagnitude = self._powerlimits[0]
        return
    vmin, vmax = sorted(self.axis.get_view_interval())
    locs = np.asarray(self.locs)
    locs = locs[(vmin <= locs) & (locs <= vmax)]
    locs = np.abs(locs)
    if not len(locs):
        self.orderOfMagnitude = 0
        return
    if self.offset:
        oom = math.floor(math.log10(vmax - vmin))
    else:
        val = locs.max()
        if val == 0:
            oom = 0
        else:
            oom = math.floor(math.log10(val))
    if oom <= self._powerlimits[0]:
        self.orderOfMagnitude = oom
    elif oom >= self._powerlimits[1]:
        self.orderOfMagnitude = oom
    else:
        self.orderOfMagnitude = 0