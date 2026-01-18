from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class PageNumberObject:

    def __init__(self, example='XXX'):
        self.example = example

    def width(self, engine):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        return stringWidth(self.example, engine.fontName, engine.fontSize)

    def execute(self, engine, textobject, canvas):
        n = canvas.getPageNumber()
        textobject.textOut(str(n))