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
class Ellipse(SolidShape):
    _attrMap = AttrMap(BASE=SolidShape, cx=AttrMapValue(isNumber, desc='x of the centre'), cy=AttrMapValue(isNumber, desc='y of the centre'), rx=AttrMapValue(isNumber, desc='x radius'), ry=AttrMapValue(isNumber, desc='y radius'))

    def __init__(self, cx, cy, rx, ry, **kw):
        SolidShape.__init__(self, kw)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry

    def copy(self):
        new = self.__class__(self.cx, self.cy, self.rx, self.ry)
        new.setProperties(self.getProperties())
        return new

    def getBounds(self):
        return (self.cx - self.rx, self.cy - self.ry, self.cx + self.rx, self.cy + self.ry)