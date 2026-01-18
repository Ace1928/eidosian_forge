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
def _yieldBLParaWords(blPara, start, stop):
    R = []
    aR = R.append
    for l in blPara.lines[start:stop]:
        for w in l[1]:
            if isinstance(w, _SplitWord):
                aR(w)
                if isinstance(w, _SplitWordEnd):
                    yield _rejoinSplitWords(R)
                    del R[:]
                continue
            elif R:
                yield _rejoinSplitWords(R)
                del R[:]
            yield w
    if R:
        yield _rejoinSplitWords(R)