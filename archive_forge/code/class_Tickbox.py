from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class Tickbox(_Symbol):
    """This draws a black box with a red tick in it - another 'checkbox'.

        possible attributes:
        'x', 'y', 'size', 'tickColor', 'strokeColor', 'tickwidth'

"""
    _attrMap = AttrMap(BASE=_Symbol, tickColor=AttrMapValue(isColorOrNone), tickwidth=AttrMapValue(isNumber))

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.tickColor = colors.red
        self.strokeColor = colors.black
        self.fillColor = colors.white
        self.tickwidth = 10

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        box = shapes.Rect(self.x + 1, self.y + 1, s - 2, s - 2, fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=2)
        g.add(box)
        tickLine = shapes.PolyLine(points=[self.x + s * 0.15, self.y + s * 0.35, self.x + s * 0.35, self.y + s * 0.15, self.x + s * 0.35, self.y + s * 0.15, self.x + s * 0.85, self.y + s * 0.85], fillColor=self.tickColor, strokeColor=self.tickColor, strokeWidth=self.tickwidth)
        g.add(tickLine)
        return g