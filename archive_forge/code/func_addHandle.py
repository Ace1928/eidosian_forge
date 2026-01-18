import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def addHandle(self, info, index=None):
    h = ROI.addHandle(self, info, index=index)
    h.sigRemoveRequested.connect(self.removeHandle)
    self.stateChanged(finish=True)
    return h