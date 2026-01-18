import re
import sys
import copy
import unicodedata
import reportlab.lib.sequencer
from reportlab.lib.abag import ABag
from reportlab.lib.utils import ImageReader, annotateException, encode_label, asUnicode
from reportlab.lib.colors import toColor, black
from reportlab.lib.fonts import tt2ps, ps2tt
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch,mm,cm,pica
from reportlab.rl_config import platypus_link_underline
from html.parser import HTMLParser
from html.entities import name2codepoint
def _valignpc(s):
    s = s.lower()
    if s in ('baseline', 'sub', 'super', 'top', 'text-top', 'middle', 'bottom', 'text-bottom'):
        return s
    if s.endswith('%'):
        n = _convnum(s[:-1])
        if isinstance(n, tuple):
            n = n[1]
        return _PCT(n)
    n = _num(s)
    if isinstance(n, tuple):
        n = n[1]
    return n