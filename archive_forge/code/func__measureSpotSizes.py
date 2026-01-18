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
def _measureSpotSizes(self, **kwargs):
    """Generate pairs (width, pxWidth) for spots in data"""
    styles = zip(*self._style(['size', 'pen'], **kwargs))
    if self.opts['pxMode']:
        for size, pen in styles:
            yield (0, size + pen.widthF())
    else:
        for size, pen in styles:
            if pen.isCosmetic():
                yield (size, pen.widthF())
            else:
                yield (size + pen.widthF(), 0)