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
def _split_blParaHard(blPara, start, stop):
    f = []
    lines = blPara.lines[start:stop]
    for l in lines:
        for w in l.words:
            f.append(w)
        if l is not lines[-1]:
            i = len(f) - 1
            while i >= 0 and hasattr(f[i], 'cbDefn') and (not getattr(f[i].cbDefn, 'width', 0)):
                i -= 1
            if i >= 0:
                g = f[i]
                if not g.text:
                    g.text = ' '
                elif g.text[-1] != ' ':
                    g.text += ' '
    return f