import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def _invertAxis(self, ax, inv):
    key = 'xy'[ax] + 'Inverted'
    if self.state[key] == inv:
        return
    self.state[key] = inv
    self._matrixNeedsUpdate = True
    self.updateViewRange()
    self.update()
    self.sigStateChanged.emit(self)
    if ax:
        self.sigYRangeChanged.emit(self, tuple(self.state['viewRange'][ax]))
    else:
        self.sigXRangeChanged.emit(self, tuple(self.state['viewRange'][ax]))