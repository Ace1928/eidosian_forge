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
class Path(SolidShape):
    """Path, made up of straight lines and bezier curves."""
    _attrMap = AttrMap(BASE=SolidShape, points=AttrMapValue(isListOfNumbers), operators=AttrMapValue(isListOfNumbers), isClipPath=AttrMapValue(isBoolean), autoclose=AttrMapValue(NoneOr(OneOf('svg', 'pdf'))), fillMode=AttrMapValue(OneOf(FILL_EVEN_ODD, FILL_NON_ZERO)))

    def __init__(self, points=None, operators=None, isClipPath=0, autoclose=None, fillMode=FILL_EVEN_ODD, **kw):
        SolidShape.__init__(self, kw)
        if points is None:
            points = []
        if operators is None:
            operators = []
        assert len(points) % 2 == 0, 'Point list must have even number of elements!'
        self.points = points
        self.operators = operators
        self.isClipPath = isClipPath
        self.autoclose = autoclose
        self.fillMode = fillMode

    def copy(self):
        new = self.__class__(self.points[:], self.operators[:])
        new.setProperties(self.getProperties())
        return new

    def moveTo(self, x, y):
        self.points.extend([x, y])
        self.operators.append(_MOVETO)

    def lineTo(self, x, y):
        self.points.extend([x, y])
        self.operators.append(_LINETO)

    def curveTo(self, x1, y1, x2, y2, x3, y3):
        self.points.extend([x1, y1, x2, y2, x3, y3])
        self.operators.append(_CURVETO)

    def closePath(self):
        self.operators.append(_CLOSEPATH)

    def getBounds(self):
        points = self.points
        try:
            X = []
            aX = X.append
            eX = X.extend
            Y = []
            aY = Y.append
            eY = Y.extend
            i = 0
            for op in self.operators:
                nArgs = _PATH_OP_ARG_COUNT[op]
                j = i + nArgs
                if nArgs == 2:
                    aX(points[i])
                    aY(points[i + 1])
                elif nArgs == 6:
                    x1, x2, x3 = points[i:j:2]
                    eX(_getBezierExtrema(X[-1], x1, x2, x3))
                    y1, y2, y3 = points[i + 1:j:2]
                    eY(_getBezierExtrema(Y[-1], y1, y2, y3))
                i = j
            return (min(X), min(Y), max(X), max(Y))
        except:
            return getPathBounds(points)