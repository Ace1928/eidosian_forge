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
def _rightDrawParaLineX(tx, offset, line, last=0):
    tx._x_offset = offset
    extraSpace = line.extraSpace
    simple = extraSpace > -1e-08 or getattr(line, 'preformatted', False)
    if not simple:
        nSpaces = line.wordCount + sum([_nbspCount(w.text) for w in line.words if not hasattr(w, 'cbDefn')]) - 1
        simple = nSpaces <= 0
    if simple:
        m = offset + line.extraSpace
        setXPos(tx, m)
        _putFragLine(m, tx, line, last, 'right')
    else:
        m = offset
        tx.setWordSpace(extraSpace / float(nSpaces))
        _putFragLine(m, tx, line, last, 'right')
        tx.setWordSpace(0)
    setXPos(tx, -m)