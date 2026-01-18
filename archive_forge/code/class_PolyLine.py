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
class PolyLine(LineShape):
    """Series of line segments.  Does not define a
    closed shape; never filled even if apparently joined.
    Put the numbers in the list, not two-tuples."""
    _attrMap = AttrMap(BASE=LineShape, points=AttrMapValue(isListOfNumbers, desc='list of numbers in the form x1, y1, x2, y2 ... xn, yn'))

    def __init__(self, points=[], **kw):
        LineShape.__init__(self, kw)
        points = points or []
        lenPoints = len(points)
        if lenPoints:
            if isSeq(points[0]):
                L = []
                for x, y in points:
                    L.append(x)
                    L.append(y)
                points = L
            else:
                assert len(points) % 2 == 0, 'Point list must have even number of elements!'
        self.points = points

    def copy(self):
        new = self.__class__(self.points)
        new.setProperties(self.getProperties())
        return new

    def getBounds(self):
        return getPointsBounds(self.points)