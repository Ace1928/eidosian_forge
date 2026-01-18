import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def _buildQImageBuffer(self, shape):
    self._displayBuffer = numpy.empty(shape[:2] + (4,), dtype=numpy.ubyte)
    if self._xp == getCupy():
        self._processingBuffer = self._xp.empty(shape[:2] + (4,), dtype=self._xp.ubyte)
    else:
        self._processingBuffer = self._displayBuffer
    self.qimage = None