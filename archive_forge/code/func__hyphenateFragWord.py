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
def _hyphenateFragWord(hyphenator, FW, newWidth, maxWidth, uriWasteReduce, embeddedHyphenation, hymwl=hyphenationMinWordLength):
    ww = FW[0]
    if ww == 0:
        return []
    if len(FW) == 2:
        f, s = FW[1]
        if isinstance(FW, _SplitFragLL):
            s = _SplitWordLL(s)
        R = _hyGenPair(hyphenator, s, ww, newWidth, maxWidth, f.fontName, f.fontSize, uriWasteReduce, embeddedHyphenation, hymwl)
        if R:
            jc, hylen, hw, tw, h, t = R
            return [(_SplitFragHY if jc else _SplitFragH)([hw + hylen, (f, h + jc)]), (_SplitFragHS if isinstance(FW, _HSFrag) else _SplitFrag)([tw, (f, t)])]
    else:
        R = _hyGenFragsPair(hyphenator, FW, newWidth, maxWidth, uriWasteReduce, embeddedHyphenation, hymwl)
        if R:
            jc, h, t = R
            return [(_SplitFragHY if jc else _SplitFragH)(h), (_SplitFragHS if isinstance(FW, _HSFrag) else _SplitFrag)(t)]
    return None