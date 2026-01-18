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
def _issueT1String(self, fontObj, x, y, s, textRenderMode=0):
    fc = fontObj
    code_append = self.code_append
    fontSize = self._fontSize
    fontsUsed = self._fontsUsed
    escape = self._escape
    if not isUnicode(s):
        try:
            s = s.decode('utf8')
        except UnicodeDecodeError as e:
            i, j = e.args[2:4]
            raise UnicodeDecodeError(*e.args[:4] + ('%s\n%s-->%s<--%s' % (e.args[4], s[i - 10:i], s[i:j], s[j:j + 10]),))
    for f, t in unicode2T1(s, [fontObj] + fontObj.substitutionFonts):
        if f != fc:
            psName = asNative(f.face.name)
            code_append('(%s) findfont %s scalefont setfont' % (psName, fp_str(fontSize)))
            if psName not in fontsUsed:
                fontsUsed.append(psName)
            fc = f
        self._textOut(x, y, t, textRenderMode)
        x += f.stringWidth(t.decode(f.encName), fontSize)
    if fontObj != fc:
        self._font = None
        self.setFont(fontObj.face.name, fontSize)