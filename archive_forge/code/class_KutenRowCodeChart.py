import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
class KutenRowCodeChart(CodeChartBase):
    """Formats one 'row' of the 94x94 space used in many Asian encodings.aliases

    These deliberately resemble the code charts in Ken Lunde's "Understanding
    CJKV Information Processing", to enable manual checking.  Due to the large
    numbers of characters, we don't try to make one graphic with 10,000 characters,
    but rather output a sequence of these."""

    def __init__(self, row, faceName, encodingName):
        self.row = row
        self.codePoints = 94
        self.boxSize = 18
        self.charsPerRow = 20
        self.rows = 5
        self.rowLabels = ['00', '20', '40', '60', '80']
        self.hex = 0
        self.faceName = faceName
        self.encodingName = encodingName
        try:
            font = cidfonts.CIDFont(self.faceName, self.encodingName)
            pdfmetrics.registerFont(font)
        except:
            self.faceName = 'Helvetica'
            self.encodingName = 'WinAnsiEncoding'
        self.fontName = self.faceName + '-' + self.encodingName
        self.calcLayout()

    def makeRow(self, row):
        """Works out the character values for this kuten row"""
        cells = []
        if self.encodingName.find('EUC') > -1:
            for col in range(1, 95):
                ch = int2Byte(row + 160) + int2Byte(col + 160)
                cells.append(ch)
        else:
            cells.append([None] * 94)
        return cells

    def draw(self):
        self.drawLabels(topLeft='R%d' % self.row)
        charList = [None] + self.makeRow(self.row)
        self.drawChars(charList)
        self.canv.grid(self.xlist, self.ylist)