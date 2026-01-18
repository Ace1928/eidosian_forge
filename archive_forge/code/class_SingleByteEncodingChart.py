import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
class SingleByteEncodingChart(CodeChartBase):

    def __init__(self, faceName='Helvetica', encodingName='WinAnsiEncoding', charsPerRow=16, boxSize=14, hex=1):
        self.codePoints = 256
        self.faceName = faceName
        self.encodingName = encodingName
        self.fontName = self.faceName + '-' + self.encodingName
        self.charsPerRow = charsPerRow
        self.boxSize = boxSize
        self.hex = hex
        self.rowLabels = None
        pdfmetrics.registerFont(pdfmetrics.Font(self.fontName, self.faceName, self.encodingName))
        self.calcLayout()

    def draw(self):
        self.drawLabels()
        charList = [None] * 32 + list(map(int2Byte, list(range(32, 256))))
        encName = self.encodingName
        encName = adobe2codec.get(encName, encName)
        decoder = codecs.lookup(encName)[1]

        def decodeFunc(txt):
            if txt is None:
                return None
            else:
                return decoder(txt, errors='replace')[0]
        charList = [decodeFunc(ch) for ch in charList]
        self.drawChars(charList)
        self.canv.grid(self.xlist, self.ylist)