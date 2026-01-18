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
def _uri_split_pairs(uri):
    if isBytes(uri):
        uri = uri.decode('utf8')
    m = uri_pat.match(uri)
    if not m:
        return None
    scheme = m.group(1)
    uri = uri[len(scheme):]
    slash = u'\\' if not scheme and u'/' not in uri else u'/'
    R = ([(scheme, uri)] if scheme and uri else []) + list(_slash_parts(uri, scheme, slash))
    R.reverse()
    return R