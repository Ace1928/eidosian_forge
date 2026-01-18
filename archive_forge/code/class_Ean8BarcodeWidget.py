from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
class Ean8BarcodeWidget(Ean13BarcodeWidget):
    codeName = 'EAN8'
    _attrMap = AttrMap(BASE=Ean13BarcodeWidget, value=AttrMapValue(nDigits(7), desc='the number'))
    _start_right = 4
    _nbars = 85
    _digits = 7
    _0csw = 3
    _1csw = 1

    def _encode_left(self, s, a):
        cp = self._lhconvert[s[0]]
        _left = self._left[0]
        z = ord('0')
        for i, c in enumerate(s[0:self._start_right]):
            a(_left[ord(c) - z])

    def _short_bar(self, i):
        i += 9 - self._lquiet
        return self.humanReadable and (12 < i < 41 or 43 < i < 73)

    def _add_human_readable(self, s, gAdd):
        barWidth = self.barWidth
        fontSize = self.fontSize
        textColor = self.textColor
        fontName = self.fontName
        fth = fontSize * 1.2
        y = self.y + 0.2 * fth
        x = (26.5 - 9 + self._lquiet) * barWidth
        c = s[0:4]
        gAdd(String(x, y, c, fontName=fontName, fontSize=fontSize, fillColor=textColor, textAnchor='middle'))
        x = (59.5 - 9 + self._lquiet) * barWidth
        c = s[4:]
        gAdd(String(x, y, c, fontName=fontName, fontSize=fontSize, fillColor=textColor, textAnchor='middle'))