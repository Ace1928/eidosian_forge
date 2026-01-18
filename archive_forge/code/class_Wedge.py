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
class Wedge(SolidShape):
    """A "slice of a pie" by default translates to a polygon moves anticlockwise
       from start angle to end angle"""
    _attrMap = AttrMap(BASE=SolidShape, centerx=AttrMapValue(isNumber, desc='x of the centre'), centery=AttrMapValue(isNumber, desc='y of the centre'), radius=AttrMapValue(isNumber, desc='radius in points'), startangledegrees=AttrMapValue(isNumber), endangledegrees=AttrMapValue(isNumber), yradius=AttrMapValue(isNumberOrNone), radius1=AttrMapValue(isNumberOrNone), yradius1=AttrMapValue(isNumberOrNone), annular=AttrMapValue(isBoolean, desc='treat as annular ring'))
    degreedelta = 1

    def __init__(self, centerx, centery, radius, startangledegrees, endangledegrees, yradius=None, annular=False, **kw):
        SolidShape.__init__(self, kw)
        while endangledegrees < startangledegrees:
            endangledegrees = endangledegrees + 360
        self.centerx, self.centery, self.radius, self.startangledegrees, self.endangledegrees = (centerx, centery, radius, startangledegrees, endangledegrees)
        self.yradius = yradius
        self.annular = annular

    def _xtraRadii(self):
        yradius = getattr(self, 'yradius', None)
        if yradius is None:
            yradius = self.radius
        radius1 = getattr(self, 'radius1', None)
        yradius1 = getattr(self, 'yradius1', radius1)
        if radius1 is None:
            radius1 = yradius1
        return (yradius, radius1, yradius1)

    def asPolygon(self):
        centerx = self.centerx
        centery = self.centery
        radius = self.radius
        yradius, radius1, yradius1 = self._xtraRadii()
        startangledegrees = self.startangledegrees
        endangledegrees = self.endangledegrees
        degreestoradians = pi / 180.0
        startangle = startangledegrees * degreestoradians
        endangle = endangledegrees * degreestoradians
        while endangle < startangle:
            endangle = endangle + 2 * pi
        angle = float(endangle - startangle)
        points = []
        if angle > 0.001:
            degreedelta = min(self.degreedelta or 1.0, angle)
            radiansdelta = degreedelta * degreestoradians
            n = max(1, int(angle / radiansdelta + 0.5))
            radiansdelta = angle / n
            n += 1
        else:
            n = 1
            radiansdelta = 0
        CA = []
        CAA = CA.append
        a = points.append
        for angle in range(n):
            angle = startangle + angle * radiansdelta
            CAA((cos(angle), sin(angle)))
        for c, s in CA:
            a(centerx + radius * c)
            a(centery + yradius * s)
        if (radius1 == 0 or radius1 is None) and (yradius1 == 0 or yradius1 is None):
            a(centerx)
            a(centery)
        else:
            CA.reverse()
            for c, s in CA:
                a(centerx + radius1 * c)
                a(centery + yradius1 * s)
        if self.annular:
            P = Path(fillMode=getattr(self, 'fillMode', FILL_EVEN_ODD))
            P.moveTo(points[0], points[1])
            for x in range(2, 2 * n, 2):
                P.lineTo(points[x], points[x + 1])
            P.closePath()
            P.moveTo(points[2 * n], points[2 * n + 1])
            for x in range(2 * n + 2, 4 * n, 2):
                P.lineTo(points[x], points[x + 1])
            P.closePath()
            return P
        else:
            return Polygon(points)

    def copy(self):
        new = self.__class__(self.centerx, self.centery, self.radius, self.startangledegrees, self.endangledegrees)
        new.setProperties(self.getProperties())
        return new

    def getBounds(self):
        return self.asPolygon().getBounds()