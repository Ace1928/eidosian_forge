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
class Line(LineShape):
    _attrMap = AttrMap(BASE=LineShape, x1=AttrMapValue(isNumber, desc=''), y1=AttrMapValue(isNumber, desc=''), x2=AttrMapValue(isNumber, desc=''), y2=AttrMapValue(isNumber, desc=''))

    def __init__(self, x1, y1, x2, y2, **kw):
        LineShape.__init__(self, kw)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def getBounds(self):
        """Returns bounding rectangle of object as (x1,y1,x2,y2)"""
        return (self.x1, self.y1, self.x2, self.y2)