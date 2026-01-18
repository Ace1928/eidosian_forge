from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
class ISBNBarcodeWidget(Ean13BarcodeWidget):
    """
    ISBN Barcodes optionally print the EAN-5 supplemental price
    barcode (with the price in dollars or pounds). Set price to a string
    that follows the EAN-5 for ISBN spec:

        leading digit 0, 1 = GBP
                      3    = AUD
                      4    = NZD
                      5    = USD
                      6    = CAD
        next 4 digits = price between 00.00 and 99.98, i.e.:

        price='52499' # $24.99 USD
    """
    codeName = 'ISBN'
    _attrMap = AttrMap(BASE=Ean13BarcodeWidget, price=AttrMapValue(NoneOr(nDigits(5)), desc='None or the price to display'))

    def draw(self):
        g = Ean13BarcodeWidget.draw(self)
        price = getattr(self, 'price', None)
        if not price:
            return g
        bounds = g.getBounds()
        x = bounds[2]
        pricecode = Ean5BarcodeWidget(x=x, value=price, price=True, humanReadable=True, barHeight=self.barHeight, quiet=self.quiet)
        g.add(pricecode)
        return g

    def _add_human_readable(self, s, gAdd):
        Ean13BarcodeWidget._add_human_readable(self, s, gAdd)
        barWidth = self.barWidth
        barHeight = self.barHeight
        fontSize = self.fontSize
        textColor = self.textColor
        fontName = self.fontName
        fth = fontSize * 1.2
        y = self.y + 0.2 * fth + barHeight
        x = self._lquiet * barWidth
        isbn = 'ISBN '
        segments = [s[0:3], s[3:4], s[4:9], s[9:12], s[12]]
        isbn += '-'.join(segments)
        gAdd(String(x, y, isbn, fontName=fontName, fontSize=fontSize, fillColor=textColor))