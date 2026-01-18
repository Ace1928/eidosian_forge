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
class LineShape(Shape):
    _attrMap = AttrMap(strokeColor=AttrMapValue(isColorOrNone), strokeWidth=AttrMapValue(isNumber), strokeLineCap=AttrMapValue(OneOf(0, 1, 2), desc='Line cap 0=butt, 1=round & 2=square'), strokeLineJoin=AttrMapValue(OneOf(0, 1, 2), desc='Line join 0=miter, 1=round & 2=bevel'), strokeMiterLimit=AttrMapValue(isNumber, desc='miter limit control miter line joins'), strokeDashArray=AttrMapValue(isStrokeDashArray, desc='[numbers] or [phase,[numbers]]'), strokeOpacity=AttrMapValue(isOpacity, desc='The level of transparency of the line, any real number betwen 0 and 1'), strokeOverprint=AttrMapValue(isBoolean, desc='Turn on stroke overprinting'), overprintMask=AttrMapValue(isBoolean, desc='overprinting for ordinary CMYK', advancedUsage=1))

    def __init__(self, kw):
        self.strokeColor = STATE_DEFAULTS['strokeColor']
        self.strokeWidth = 1
        self.strokeLineCap = 0
        self.strokeLineJoin = 0
        self.strokeMiterLimit = 0
        self.strokeDashArray = None
        self.strokeOpacity = None
        self.setProperties(kw)