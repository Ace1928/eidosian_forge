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
def _pinv_fallback(tr):
    arr = np.array([tr.m11(), tr.m12(), tr.m13(), tr.m21(), tr.m22(), tr.m23(), tr.m31(), tr.m32(), tr.m33()])
    arr.shape = (3, 3)
    pinv = np.linalg.pinv(arr)
    return QtGui.QTransform(*pinv.ravel().tolist())