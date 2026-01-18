import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class SegmentToPointPen(AbstractPen):
    """
    Adapter class that converts the (Segment)Pen protocol to the
    PointPen protocol.
    """

    def __init__(self, pointPen, guessSmooth=True):
        if guessSmooth:
            self.pen = GuessSmoothPointPen(pointPen)
        else:
            self.pen = pointPen
        self.contour = None

    def _flushContour(self):
        pen = self.pen
        pen.beginPath()
        for pt, segmentType in self.contour:
            pen.addPoint(pt, segmentType=segmentType)
        pen.endPath()

    def moveTo(self, pt):
        self.contour = []
        self.contour.append((pt, 'move'))

    def lineTo(self, pt):
        if self.contour is None:
            raise PenError('Contour missing required initial moveTo')
        self.contour.append((pt, 'line'))

    def curveTo(self, *pts):
        if not pts:
            raise TypeError('Must pass in at least one point')
        if self.contour is None:
            raise PenError('Contour missing required initial moveTo')
        for pt in pts[:-1]:
            self.contour.append((pt, None))
        self.contour.append((pts[-1], 'curve'))

    def qCurveTo(self, *pts):
        if not pts:
            raise TypeError('Must pass in at least one point')
        if pts[-1] is None:
            self.contour = []
        elif self.contour is None:
            raise PenError('Contour missing required initial moveTo')
        for pt in pts[:-1]:
            self.contour.append((pt, None))
        if pts[-1] is not None:
            self.contour.append((pts[-1], 'qcurve'))

    def closePath(self):
        if self.contour is None:
            raise PenError('Contour missing required initial moveTo')
        if len(self.contour) > 1 and self.contour[0][0] == self.contour[-1][0]:
            self.contour[0] = self.contour[-1]
            del self.contour[-1]
        else:
            pt, tp = self.contour[0]
            if tp == 'move':
                self.contour[0] = (pt, 'line')
        self._flushContour()
        self.contour = None

    def endPath(self):
        if self.contour is None:
            raise PenError('Contour missing required initial moveTo')
        self._flushContour()
        self.contour = None

    def addComponent(self, glyphName, transform):
        if self.contour is not None:
            raise PenError('Components must be added before or after contours')
        self.pen.addComponent(glyphName, transform)