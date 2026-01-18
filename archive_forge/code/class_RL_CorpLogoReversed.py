from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
class RL_CorpLogoReversed(RL_CorpLogo):

    def __init__(self):
        RL_CorpLogo.__init__(self)
        self.background = white
        self.fillColor = ReportLabBlue