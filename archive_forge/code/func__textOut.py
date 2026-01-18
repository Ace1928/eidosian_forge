from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def _textOut(self, text, TStar=0):
    """prints string at current point, ignores text cursor"""
    self._code.append('%s%s' % (self._formatText(text), TStar and ' T*' or ''))