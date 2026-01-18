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
def _split_blParaProcessed(self, blPara, start, stop):
    if not stop:
        return []
    lines = blPara.lines
    sFW = lines[start].sFW
    sFWN = lines[stop].sFW if stop != len(lines) else len(self.frags)
    F = self.frags[sFW:sFWN]
    while F and isinstance(F[-1], _InjectedFrag):
        del F[-1]
    if isinstance(F[-1], _SplitFragHY):
        F[-1].__class__ = _SHYWordHS if isinstance(F[-1], _SHYWord) else _SplitFragLL
    return F