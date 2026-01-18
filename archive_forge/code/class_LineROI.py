import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class LineROI(ROI):
    """
    Rectangular ROI subclass with scale-rotate handles on either side. This
    allows the ROI to be positioned as if moving the ends of a line segment.
    A third handle controls the width of the ROI orthogonal to its "line" axis.
    
    ============== =============================================================
    **Arguments**
    pos1           (length-2 sequence) The position of the center of the ROI's
                   left edge.
    pos2           (length-2 sequence) The position of the center of the ROI's
                   right edge.
    width          (float) The width of the ROI.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos1, pos2, width, **args):
        pos1 = Point(pos1)
        pos2 = Point(pos2)
        d = pos2 - pos1
        l = d.length()
        ra = d.angle(Point(1, 0), units='radians')
        c = Point(width / 2.0 * sin(ra), -width / 2.0 * cos(ra))
        pos1 = pos1 + c
        ROI.__init__(self, pos1, size=Point(l, width), angle=degrees(ra), **args)
        self.addScaleRotateHandle([0, 0.5], [1, 0.5])
        self.addScaleRotateHandle([1, 0.5], [0, 0.5])
        self.addScaleHandle([0.5, 1], [0.5, 0.5])