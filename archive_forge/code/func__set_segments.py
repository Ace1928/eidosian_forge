import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
def _set_segments(glyph, segments, reverse_direction):
    """Draw segments as extracted by GetSegmentsPen back to a glyph."""
    glyph.clearContours()
    pen = glyph.getPen()
    if reverse_direction:
        pen = ReverseContourPen(pen)
    for tag, args in segments:
        if tag == 'move':
            pen.moveTo(*args)
        elif tag == 'line':
            pen.lineTo(*args)
        elif tag == 'curve':
            pen.curveTo(*args[1:])
        elif tag == 'qcurve':
            pen.qCurveTo(*args[1:])
        elif tag == 'close':
            pen.closePath()
        elif tag == 'end':
            pen.endPath()
        else:
            raise AssertionError('Unhandled segment type "%s"' % tag)