from math import atan2, degrees
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
def _posToRel(self, pos):
    pt1, pt2 = self.getEndpoints()
    if pt1 is None:
        return 0
    pos = self.mapToParent(pos)
    return (pos.x() - pt1.x()) / (pt2.x() - pt1.x())