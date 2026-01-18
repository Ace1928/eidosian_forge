from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def _checkSeparation(self, cmyk):
    if isinstance(cmyk, CMYKColorSep):
        name, sname = self._doc.addColor(cmyk)
        if name not in self._colorsUsed:
            self._colorsUsed[name] = sname
        return name