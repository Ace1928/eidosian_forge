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
def _xtraRadii(self):
    yradius = getattr(self, 'yradius', None)
    if yradius is None:
        yradius = self.radius
    radius1 = getattr(self, 'radius1', None)
    yradius1 = getattr(self, 'yradius1', radius1)
    if radius1 is None:
        radius1 = yradius1
    return (yradius, radius1, yradius1)