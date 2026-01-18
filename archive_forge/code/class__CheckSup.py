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
class _CheckSup(_ExValidate):
    """class for syntax checking <sup|sub> attributes
    if the check succeeds then we always return the string for later evaluation"""

    def validate(self, parser, s):
        self.fontSize = parser._stack[-1].fontSize
        fontSizeNormalize(self, self.attr, '')
        return s

    def __call__(self, parser, s):
        setattr(self, self.attr, s)
        return _ExValidate.__call__(self, parser, s)