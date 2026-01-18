from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _jtimes(v, Jv, x, y, fy=None):
    yv = np.concatenate((y, v))
    if len(_p) > 0:
        Jv[:] = np.asarray(self.jtimes_cb(x, yv, _p))
    else:
        Jv[:] = np.asarray(self.jtimes_cb(x, yv))