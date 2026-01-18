from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
class ReportLabLogo:
    """vector reportlab logo centered in a 250x by 150y rectangle"""

    def __init__(self, atx=0, aty=0, width=2.5 * inch, height=1.5 * inch, powered_by=0):
        self.origin = (atx, aty)
        self.dimensions = (width, height)
        self.powered_by = powered_by

    def draw(self, canvas):
        from reportlab.graphics import renderPDF
        canvas.saveState()
        atx, aty = self.origin
        width, height = self.dimensions
        logo = RL_CorpLogo()
        logo.width, logo.height = (width, height)
        renderPDF.draw(logo.demo(), canvas, atx, aty, 0)
        canvas.restoreState()