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
def _num(s, unit=1, allowRelative=True, _unit_map={'i': inch, 'in': inch, 'pt': 1, 'cm': cm, 'mm': mm, 'pica': pica}, _re_unit=re.compile('^\\s*(.*)(i|in|cm|mm|pt|pica)\\s*$')):
    """Convert a string like '10cm' to an int or float (in points).
       The default unit is point, but optionally you can use other
       default units like mm.
    """
    m = _re_unit.match(s)
    if m:
        unit = _unit_map[m.group(2)]
        s = m.group(1)
    return _convnum(s, unit, allowRelative)