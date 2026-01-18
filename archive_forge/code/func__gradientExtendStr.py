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
def _gradientExtendStr(extend):
    if isinstance(extend, (list, tuple)):
        if len(extend) != 2:
            raise ValueError('wrong length for extend argument' % extend)
        return '[%s %s]' % ['true' if _ else 'false' for _ in extend]
    return '[true true]' if extend else '[false false]'