import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class BasePointToSegmentPen(AbstractPointPen):
    """
    Base class for retrieving the outline in a segment-oriented
    way. The PointPen protocol is simple yet also a little tricky,
    so when you need an outline presented as segments but you have
    as points, do use this base implementation as it properly takes
    care of all the edge cases.
    """

    def __init__(self):
        self.currentPath = None

    def beginPath(self, identifier=None, **kwargs):
        if self.currentPath is not None:
            raise PenError('Path already begun.')
        self.currentPath = []

    def _flushContour(self, segments):
        """Override this method.

        It will be called for each non-empty sub path with a list
        of segments: the 'segments' argument.

        The segments list contains tuples of length 2:
                (segmentType, points)

        segmentType is one of "move", "line", "curve" or "qcurve".
        "move" may only occur as the first segment, and it signifies
        an OPEN path. A CLOSED path does NOT start with a "move", in
        fact it will not contain a "move" at ALL.

        The 'points' field in the 2-tuple is a list of point info
        tuples. The list has 1 or more items, a point tuple has
        four items:
                (point, smooth, name, kwargs)
        'point' is an (x, y) coordinate pair.

        For a closed path, the initial moveTo point is defined as
        the last point of the last segment.

        The 'points' list of "move" and "line" segments always contains
        exactly one point tuple.
        """
        raise NotImplementedError

    def endPath(self):
        if self.currentPath is None:
            raise PenError('Path not begun.')
        points = self.currentPath
        self.currentPath = None
        if not points:
            return
        if len(points) == 1:
            pt, segmentType, smooth, name, kwargs = points[0]
            segments = [('move', [(pt, smooth, name, kwargs)])]
            self._flushContour(segments)
            return
        segments = []
        if points[0][1] == 'move':
            pt, segmentType, smooth, name, kwargs = points[0]
            segments.append(('move', [(pt, smooth, name, kwargs)]))
            points.pop(0)
        else:
            firstOnCurve = None
            for i in range(len(points)):
                segmentType = points[i][1]
                if segmentType is not None:
                    firstOnCurve = i
                    break
            if firstOnCurve is None:
                points.append((None, 'qcurve', None, None, None))
            else:
                points = points[firstOnCurve + 1:] + points[:firstOnCurve + 1]
        currentSegment = []
        for pt, segmentType, smooth, name, kwargs in points:
            currentSegment.append((pt, smooth, name, kwargs))
            if segmentType is None:
                continue
            segments.append((segmentType, currentSegment))
            currentSegment = []
        self._flushContour(segments)

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs):
        if self.currentPath is None:
            raise PenError('Path not begun')
        self.currentPath.append((pt, segmentType, smooth, name, kwargs))