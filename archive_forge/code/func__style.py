import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _style(self, opts, data=None, idx=None, scale=None):
    if data is None:
        data = self.data
    if idx is None:
        idx = np.s_[:]
    for opt in opts:
        col = data[opt][idx]
        if col.base is not None:
            col = col.copy()
        if self.opts['hoverable']:
            val = self.opts['hover' + opt.title()]
            if val != _DEFAULT_STYLE[opt]:
                col[data['hovered'][idx]] = val
        col[np.equal(col, _DEFAULT_STYLE[opt])] = self.opts[opt]
        if opt == 'size' and scale is not None:
            col *= scale
        yield col