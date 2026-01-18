import itertools
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Group, Rect
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColor, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.units import mm
from reportlab.lib.utils import asUnicodeEx, isUnicode
from reportlab.graphics.barcode import qrencoder
class QrCode(Flowable):
    height = 32 * mm
    width = 32 * mm
    qrBorder = 4
    qrLevel = 'L'
    qrVersion = None
    value = None

    def __init__(self, value=None, **kw):
        self.value = isUnicodeOrQRList.normalize(value)
        for k, v in kw.items():
            setattr(self, k, v)
        ec_level = getattr(qrencoder.QRErrorCorrectLevel, self.qrLevel)
        self.qr = qrencoder.QRCode(self.qrVersion, ec_level)
        if isUnicode(self.value):
            self.addData(self.value)
        elif self.value:
            for v in self.value:
                self.addData(v)

    def addData(self, value):
        self.qr.addData(value)

    def draw(self):
        self.qr.make()
        moduleCount = self.qr.getModuleCount()
        border = self.qrBorder
        xsize = self.width / (moduleCount + border * 2.0)
        ysize = self.height / (moduleCount + border * 2.0)
        for r, row in enumerate(self.qr.modules):
            row = map(bool, row)
            c = 0
            for t, tt in itertools.groupby(row):
                isDark = t
                count = len(list(tt))
                if isDark:
                    x = (c + border) * xsize
                    y = self.height - (r + border + 1) * ysize
                    self.rect(x, y, count * xsize, ysize * 1.05)
                c += count

    def rect(self, x, y, w, h):
        self.canv.rect(x, y, w, h, stroke=0, fill=1)