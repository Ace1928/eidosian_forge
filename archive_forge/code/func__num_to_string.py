import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _num_to_string(self, x, vmin, vmax):
    fx = math.log(x) / math.log(self._base)
    if abs(fx) > 10000:
        s = '%1.0g' % fx
    elif abs(fx) < 1:
        s = '%1.0g' % fx
    else:
        fd = math.log(vmax - vmin) / math.log(self._base)
        s = self._pprint_val(fx, fd)
    return s