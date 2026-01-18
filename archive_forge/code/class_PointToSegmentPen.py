import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class PointToSegmentPen(BasePointToSegmentPen):
    """
    Adapter class that converts the PointPen protocol to the
    (Segment)Pen protocol.

    NOTE: The segment pen does not support and will drop point names, identifiers
    and kwargs.
    """

    def __init__(self, segmentPen, outputImpliedClosingLine=False):
        BasePointToSegmentPen.__init__(self)
        self.pen = segmentPen
        self.outputImpliedClosingLine = outputImpliedClosingLine

    def _flushContour(self, segments):
        if not segments:
            raise PenError('Must have at least one segment.')
        pen = self.pen
        if segments[0][0] == 'move':
            closed = False
            points = segments[0][1]
            if len(points) != 1:
                raise PenError(f'Illegal move segment point count: {len(points)}')
            movePt, _, _, _ = points[0]
            del segments[0]
        else:
            closed = True
            segmentType, points = segments[-1]
            movePt, _, _, _ = points[-1]
        if movePt is None:
            pass
        else:
            pen.moveTo(movePt)
        outputImpliedClosingLine = self.outputImpliedClosingLine
        nSegments = len(segments)
        lastPt = movePt
        for i in range(nSegments):
            segmentType, points = segments[i]
            points = [pt for pt, _, _, _ in points]
            if segmentType == 'line':
                if len(points) != 1:
                    raise PenError(f'Illegal line segment point count: {len(points)}')
                pt = points[0]
                if i + 1 != nSegments or outputImpliedClosingLine or (not closed) or (pt == lastPt):
                    pen.lineTo(pt)
                    lastPt = pt
            elif segmentType == 'curve':
                pen.curveTo(*points)
                lastPt = points[-1]
            elif segmentType == 'qcurve':
                pen.qCurveTo(*points)
                lastPt = points[-1]
            else:
                raise PenError(f'Illegal segmentType: {segmentType}')
        if closed:
            pen.closePath()
        else:
            pen.endPath()

    def addComponent(self, glyphName, transform, identifier=None, **kwargs):
        del identifier
        del kwargs
        self.pen.addComponent(glyphName, transform)