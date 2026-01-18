import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def _linkToView_internal(self, view):
    self.unlinkFromView()
    self._linkedView = weakref.ref(view)
    if self.orientation in ['right', 'left']:
        view.sigYRangeChanged.connect(self.linkedViewChanged)
    else:
        view.sigXRangeChanged.connect(self.linkedViewChanged)
    view.sigResized.connect(self.linkedViewChanged)