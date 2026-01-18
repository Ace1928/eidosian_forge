import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
def _get_segments(glyph):
    """Get a glyph's segments as extracted by GetSegmentsPen."""
    pen = GetSegmentsPen()
    pointPen = PointToSegmentPen(pen, outputImpliedClosingLine=True)
    glyph.drawPoints(pointPen)
    return pen.segments