import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
def _t1_re_encode(self):
    if not self._fontsUsed:
        return
    C = []
    for fontName in self._fontsUsed:
        fontObj = getFont(fontName)
        if not fontObj._dynamicFont and fontObj.encName == 'WinAnsiEncoding':
            C.append('WinAnsiEncoding /%s /%s RE' % (fontName, fontName))
    if C:
        C.insert(0, PS_WinAnsiEncoding)
        self.code.insert(1, self._sep.join(C))