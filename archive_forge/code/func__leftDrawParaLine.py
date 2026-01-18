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
def _leftDrawParaLine(tx, offset, extraspace, words, last=0):
    simple = extraspace > -1e-08 or getattr(tx, 'preformatted', False)
    text = ' '.join(words)
    setXPos(tx, offset)
    if not simple:
        nSpaces = len(words) + _nbspCount(text) - 1
        simple = nSpaces <= 0
    if simple:
        tx._textOut(text, 1)
    else:
        tx.setWordSpace(extraspace / float(nSpaces))
        tx._textOut(text, 1)
        tx.setWordSpace(0)
    setXPos(tx, -offset)
    return offset