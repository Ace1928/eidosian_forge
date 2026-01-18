import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
def _getBezierExtrema(y0, y1, y2, y3):
    """
    this is used to find if a curveTo path operator has extrema in its range
    The curveTo operator is defined by the points y0, y1, y2, y3

        B(t):=(1-t)^3*y0+3*(1-t)^2*t*y1+3*(1-t)*t^2*y2+t^3*y3
            :=t^3*(y3-3*y2+3*y1-y0)+t^2*(3*y2-6*y1+3*y0)+t*(3*y1-3*y0)+y0
    and is a cubic bezier curve.

    The differential is a quadratic
        t^2*(3*y3-9*y2+9*y1-3*y0)+t*(6*y2-12*y1+6*y0)+3*y1-3*y0

    The extrema must be at real roots, r, of the above which lie in 0<=r<=1

    The quadratic coefficients are
        a=3*y3-9*y2+9*y1-3*y0 b=6*y2-12*y1+6*y0 c=3*y1-3*y0
    or
        a=y3-3*y2+3*y1-y0 b=2*y2-4*y1+2*y0 c=y1-y0  (remove common factor of 3)
    or
        a=y3-3*(y2-y1)-y0 b=2*(y2-2*y1+y0) c=y1-y0

    The returned value is [y0,x1,x2,y3] where if found x1, x2 are any extremals that were found;
    there can be 0, 1 or 2 extremals
    """
    a = y3 - 3 * (y2 - y1) - y0
    b = 2 * (y2 - 2 * y1 + y0)
    c = y1 - y0
    Y = [y0]
    d = b * b - 4 * a * c
    if d >= 0:
        d = sqrt(d)
        if b < 0:
            d = -d
        q = -0.5 * (b + d)
        R = []
        try:
            R.append(q / a)
        except:
            pass
        try:
            R.append(c / q)
        except:
            pass
        b *= 1.5
        c *= 3
        for t in R:
            if 0 <= t <= 1:
                Y.append(t * (t * (t * a + b) + c) + y0)
    Y.append(y3)
    return Y