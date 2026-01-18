import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def absolutePosition(self, x, y):
    """return the absolute position of x,y in user space w.r.t. default user space"""
    if not ENABLE_TRACKING:
        raise ValueError('tracking not enabled! (canvas.ENABLE_TRACKING=0)')
    a, b, c, d, e, f = self._currentMatrix
    xp = a * x + c * y + e
    yp = b * x + d * y + f
    return (xp, yp)