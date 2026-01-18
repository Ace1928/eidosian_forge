import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
class GetSegmentsPen(AbstractPen):
    """Pen to collect segments into lists of points for conversion.

    Curves always include their initial on-curve point, so some points are
    duplicated between segments.
    """

    def __init__(self):
        self._last_pt = None
        self.segments = []

    def _add_segment(self, tag, *args):
        if tag in ['move', 'line', 'qcurve', 'curve']:
            self._last_pt = args[-1]
        self.segments.append((tag, args))

    def moveTo(self, pt):
        self._add_segment('move', pt)

    def lineTo(self, pt):
        self._add_segment('line', pt)

    def qCurveTo(self, *points):
        self._add_segment('qcurve', self._last_pt, *points)

    def curveTo(self, *points):
        self._add_segment('curve', self._last_pt, *points)

    def closePath(self):
        self._add_segment('close')

    def endPath(self):
        self._add_segment('end')

    def addComponent(self, glyphName, transformation):
        pass