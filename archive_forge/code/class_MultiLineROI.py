import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class MultiLineROI(MultiRectROI):

    def __init__(self, *args, **kwds):
        MultiRectROI.__init__(self, *args, **kwds)
        print('Warning: MultiLineROI has been renamed to MultiRectROI. (and MultiLineROI may be redefined in the future)')