from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _compute_backfill_indices(isfinite):
    mask = ~isfinite
    idx = np.arange(len(isfinite))
    idx[mask] = -1
    np.maximum.accumulate(idx, out=idx)
    first = np.searchsorted(idx, 0)
    if first < len(isfinite):
        idx[:first] = first
        return idx
    else:
        return None