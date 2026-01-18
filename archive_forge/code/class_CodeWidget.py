import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
class CodeWidget(Widget):
    """Block showing all the characters"""

    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 160
        self.height = 160

    def draw(self):
        dx = self.width / 16.0
        dy = self.height / 16.0
        g = Group()
        g.add(Rect(self.x, self.y, self.width, self.height, fillColor=None, strokeColor=colors.black))
        for x in range(16):
            for y in range(16):
                charValue = y * 16 + x
                if charValue > 32:
                    s = String(self.x + x * dx, self.y + (self.height - y * dy), int2Byte(charValue))
                    g.add(s)
        return g