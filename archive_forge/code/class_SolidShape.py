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
class SolidShape(LineShape):
    _attrMap = AttrMap(BASE=LineShape, fillColor=AttrMapValue(isColorOrNone, desc='filling color of the shape, e.g. red'), fillOpacity=AttrMapValue(isOpacity, desc='the level of transparency of the color, any real number between 0 and 1'), fillOverprint=AttrMapValue(isBoolean, desc='Turn on fill overprinting'), overprintMask=AttrMapValue(isBoolean, desc='overprinting for ordinary CMYK', advancedUsage=1), fillMode=AttrMapValue(OneOf(FILL_EVEN_ODD, FILL_NON_ZERO)))

    def __init__(self, kw):
        self.fillColor = STATE_DEFAULTS['fillColor']
        self.fillOpacity = None
        LineShape.__init__(self, kw)