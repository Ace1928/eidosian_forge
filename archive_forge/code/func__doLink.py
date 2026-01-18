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
def _doLink(tx, link, rect):
    if not link:
        return
    if link.startswith('#'):
        tx._canvas.linkRect('', link[1:], rect, relative=1)
    else:
        parts = link.split(':', 1)
        scheme = len(parts) == 2 and parts[0].lower() or ''
        if scheme == 'document':
            tx._canvas.linkRect('', parts[1], rect, relative=1)
        elif _scheme_re.match(scheme):
            kind = scheme.lower() == 'pdf' and 'GoToR' or 'URI'
            if kind == 'GoToR':
                link = parts[1]
            tx._canvas.linkURL(link, rect, relative=1, kind=kind)
        else:
            tx._canvas.linkURL(link, rect, relative=1, kind='URI')