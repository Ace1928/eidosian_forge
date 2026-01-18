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
def _do_under_line(i, x1, ws, tx, us_lines):
    xs = tx.XtraState
    style = xs.style
    y0 = xs.cur_y - i * style.leading
    f = xs.f
    fs = f.fontSize
    tc = f.textColor
    values = dict(L=fs, F=fs, f=fs)
    dw = tx._defaultLineWidth
    x2 = x1 + tx._canvas.stringWidth(' '.join(tx.XtraState.lines[i][1]), tx._fontname, fs) + ws
    for n, k, c, w, o, r, m, g in us_lines:
        underline = k == 'underline'
        lw = _usConv(w, values, default=tx._defaultLineWidth)
        lg = _usConv(g, values, default=1)
        dy = lg + lw
        if not underline:
            dy = -dy
        y = y0 + r + _usConv(('-0.125*L' if underline else '0.25*L') if o == '' else o, values)
        if not c:
            c = tc
        while m > 0:
            tx._do_line(x1, y, x2, y, lw, c)
            y -= dy
            m -= 1