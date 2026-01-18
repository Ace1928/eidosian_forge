import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class RulerROI(LineSegmentROI):

    def paint(self, p, *args):
        LineSegmentROI.paint(self, p, *args)
        h1 = self.handles[0]['item'].pos()
        h2 = self.handles[1]['item'].pos()
        p1 = p.transform().map(h1)
        p2 = p.transform().map(h2)
        vec = Point(h2) - Point(h1)
        length = vec.length()
        angle = vec.angle(Point(1, 0))
        pvec = p2 - p1
        pvecT = Point(pvec.y(), -pvec.x())
        pos = 0.5 * (p1 + p2) + pvecT * 40 / pvecT.length()
        p.resetTransform()
        txt = fn.siFormat(length, suffix='m') + '\n%0.1f deg' % angle
        p.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 50, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, txt)

    def boundingRect(self):
        r = LineSegmentROI.boundingRect(self)
        pxl = self.pixelLength(Point([1, 0]))
        if pxl is None:
            return r
        pxw = 50 * pxl
        return r.adjusted(-50, -50, 50, 50)