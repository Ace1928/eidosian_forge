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
def _copyNamedContents(self, obj, aKeys=None, noCopy=('contents',)):
    from copy import copy
    self_contents = self.contents
    if not aKeys:
        aKeys = list(self._attrMap.keys())
    for k, v in self.__dict__.items():
        if v in self_contents:
            pos = self_contents.index(v)
            setattr(obj, k, obj.contents[pos])
        elif k in aKeys and k not in noCopy:
            setattr(obj, k, copy(v))