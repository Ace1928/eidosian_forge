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
def _splitFragWord(w, maxWidth, maxWidths, lineno):
    """given a frag word, w, as returned by getFragWords
    split it into frag words that fit in lines of length
    maxWidth
    maxWidths[lineno+1]
    .....
    maxWidths[lineno+n]

    return the new word list which is either 
    _SplitFrag....._SPlitFrag or
    _SplitFrag....._SplitFragHS if the word is hanging space.
    """
    R = []
    maxlineno = len(maxWidths) - 1
    W = []
    lineWidth = 0
    fragText = u''
    wordWidth = 0
    f = w[1][0]
    for g, cw, c in _fragWordIter(w):
        newLineWidth = lineWidth + cw
        tooLong = newLineWidth > maxWidth
        if g is not f or tooLong:
            f = f.clone()
            if hasattr(f, 'text'):
                f.text = fragText
            W.append((f, fragText))
            if tooLong:
                W = _SplitFrag([wordWidth] + W)
                R.append(W)
                lineno += 1
                maxWidth = maxWidths[min(maxlineno, lineno)]
                W = []
                newLineWidth = cw
                wordWidth = 0
            fragText = u''
            f = g
        wordWidth += cw
        fragText += c
        lineWidth = newLineWidth
    W.append((f, fragText))
    W = (_SplitFragHS if isinstance(w, _HSFrag) else _SplitFragH)([wordWidth] + W)
    R.append(W)
    return R