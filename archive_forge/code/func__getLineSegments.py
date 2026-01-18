from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def _getLineSegments(self):
    if not self._lineSegmentsRendered:
        x, y = self.getData()
        if self.opts['stepMode']:
            x, y = self._generateStepModeData(self.opts['stepMode'], x, y, baseline=self.opts['fillLevel'])
        self._lineSegments = arrayToLineSegments(x, y, connect=self.opts['connect'], finiteCheck=not self.opts['skipFiniteCheck'], out=self._lineSegments)
        self._lineSegmentsRendered = True
    return self._lineSegments.drawargs()