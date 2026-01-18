from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
class cjkU(str):
    """simple class to hold the frag corresponding to a str"""

    def __new__(cls, value, frag, encoding):
        self = str.__new__(cls, value)
        self._frag = frag
        if hasattr(frag, 'cbDefn'):
            w = getattr(frag.cbDefn, 'width', 0)
            self._width = w
        else:
            self._width = stringWidth(value, frag.fontName, frag.fontSize)
        return self
    frag = property(lambda self: self._frag)
    width = property(lambda self: self._width)