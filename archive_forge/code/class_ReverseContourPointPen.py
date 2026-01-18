import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class ReverseContourPointPen(AbstractPointPen):
    """
    This is a PointPen that passes outline data to another PointPen, but
    reversing the winding direction of all contours. Components are simply
    passed through unchanged.

    Closed contours are reversed in such a way that the first point remains
    the first point.
    """

    def __init__(self, outputPointPen):
        self.pen = outputPointPen
        self.currentContour = None

    def _flushContour(self):
        pen = self.pen
        contour = self.currentContour
        if not contour:
            pen.beginPath(identifier=self.currentContourIdentifier)
            pen.endPath()
            return
        closed = contour[0][1] != 'move'
        if not closed:
            lastSegmentType = 'move'
        else:
            contour.append(contour.pop(0))
            firstOnCurve = None
            for i in range(len(contour)):
                if contour[i][1] is not None:
                    firstOnCurve = i
                    break
            if firstOnCurve is None:
                lastSegmentType = None
            else:
                lastSegmentType = contour[firstOnCurve][1]
        contour.reverse()
        if not closed:
            while contour[0][1] is None:
                contour.pop(0)
        pen.beginPath(identifier=self.currentContourIdentifier)
        for pt, nextSegmentType, smooth, name, kwargs in contour:
            if nextSegmentType is not None:
                segmentType = lastSegmentType
                lastSegmentType = nextSegmentType
            else:
                segmentType = None
            pen.addPoint(pt, segmentType=segmentType, smooth=smooth, name=name, **kwargs)
        pen.endPath()

    def beginPath(self, identifier=None, **kwargs):
        if self.currentContour is not None:
            raise PenError('Path already begun')
        self.currentContour = []
        self.currentContourIdentifier = identifier
        self.onCurve = []

    def endPath(self):
        if self.currentContour is None:
            raise PenError('Path not begun')
        self._flushContour()
        self.currentContour = None

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs):
        if self.currentContour is None:
            raise PenError('Path not begun')
        if identifier is not None:
            kwargs['identifier'] = identifier
        self.currentContour.append((pt, segmentType, smooth, name, kwargs))

    def addComponent(self, glyphName, transform, identifier=None, **kwargs):
        if self.currentContour is not None:
            raise PenError('Components must be added before or after contours')
        self.pen.addComponent(glyphName, transform, identifier=identifier, **kwargs)