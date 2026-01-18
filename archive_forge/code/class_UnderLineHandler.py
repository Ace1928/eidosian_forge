from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class UnderLineHandler:

    def __init__(self, color=None):
        self.color = color

    def start_at(self, x, y, para, canvas, textobject):
        self.xStart = x
        self.yStart = y

    def end_at(self, x, y, para, canvas, textobject):
        offset = para.fontSize / 8.0
        canvas.saveState()
        color = self.color
        if self.color is None:
            color = para.fontColor
        canvas.setStrokeColor(color)
        canvas.line(self.xStart, self.yStart - offset, x, y - offset)
        canvas.restoreState()