from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
class UPCA(Ean13BarcodeWidget):
    codeName = 'UPCA'
    _attrMap = AttrMap(BASE=Ean13BarcodeWidget, value=AttrMapValue(nDigits(11), desc='the number'))
    _start_right = 6
    _digits = 11
    _0csw = 3
    _1csw = 1
    _nbars = 1 + 7 * 11 + 2 * 3 + 5

    def _encode_left(self, s, a):
        cp = self._lhconvert[s[0]]
        _left = self._left[0]
        z = ord('0')
        for i, c in enumerate(s[0:self._start_right]):
            a(_left[ord(c) - z])

    def _short_bar(self, i):
        i += 9 - self._lquiet
        return self.humanReadable and (18 < i < 55 or 57 < i < 93)

    def _add_human_readable(self, s, gAdd):
        barWidth = self.barWidth
        fontSize = self.fontSize
        textColor = self.textColor
        fontName = self.fontName
        fth = fontSize * 1.2
        c = s[0]
        w = stringWidth(c, fontName, fontSize)
        x = self.x + barWidth * (self._lquiet - 8)
        y = self.y + 0.2 * fth
        gAdd(String(x, y, c, fontName=fontName, fontSize=fontSize, fillColor=textColor))
        x = self.x + (38 - 9 + self._lquiet) * barWidth
        c = s[1:6]
        gAdd(String(x, y, c, fontName=fontName, fontSize=fontSize, fillColor=textColor, textAnchor='middle'))
        x += 36 * barWidth
        c = s[6:11]
        gAdd(String(x, y, c, fontName=fontName, fontSize=fontSize, fillColor=textColor, textAnchor='middle'))
        x += 32 * barWidth
        c = s[11]
        gAdd(String(x, y, c, fontName=fontName, fontSize=fontSize, fillColor=textColor))