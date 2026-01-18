from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
def _getText(self, x=0, y=0, color=None):
    return String(x, y, self._text, fontName=self._fontName, fontSize=self._fontSize, fillColor=color)