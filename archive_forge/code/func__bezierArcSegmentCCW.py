from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def _bezierArcSegmentCCW(self, cx, cy, rx, ry, theta0, theta1):
    """compute the control points for a bezier arc with theta1-theta0 <= 90.
        Points are computed for an arc with angle theta increasing in the
        counter-clockwise (CCW) direction.  returns a tuple with starting point
        and 3 control points of a cubic bezier curve for the curvto opertator"""
    assert abs(theta1 - theta0) <= 90
    cos0 = cos(pi * theta0 / 180.0)
    sin0 = sin(pi * theta0 / 180.0)
    x0 = cx + rx * cos0
    y0 = cy + ry * sin0
    cos1 = cos(pi * theta1 / 180.0)
    sin1 = sin(pi * theta1 / 180.0)
    x3 = cx + rx * cos1
    y3 = cy + ry * sin1
    dx1 = -rx * sin0
    dy1 = ry * cos0
    halfAng = pi * (theta1 - theta0) / (2.0 * 180.0)
    k = abs(4.0 / 3.0 * (1.0 - cos(halfAng)) / sin(halfAng))
    x1 = x0 + dx1 * k
    y1 = y0 + dy1 * k
    dx2 = -rx * sin1
    dy2 = ry * cos1
    x2 = x3 - dx2 * k
    y2 = y3 - dy2 * k
    return ((x0, y0), ((x1, y1), (x2, y2), (x3, y3)))