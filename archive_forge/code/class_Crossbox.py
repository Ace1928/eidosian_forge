from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class Crossbox(_Symbol):
    """This draws a black box with a red cross in it - a 'checkbox'.

        possible attributes:
        'x', 'y', 'size', 'crossColor', 'strokeColor', 'crosswidth'

    """
    _attrMap = AttrMap(BASE=_Symbol, crossColor=AttrMapValue(isColorOrNone), crosswidth=AttrMapValue(isNumber))

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.fillColor = colors.white
        self.crossColor = colors.red
        self.strokeColor = colors.black
        self.crosswidth = 10

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        box = shapes.Rect(self.x + 1, self.y + 1, s - 2, s - 2, fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=2)
        g.add(box)
        crossLine1 = shapes.Line(self.x + s * 0.15, self.y + s * 0.15, self.x + s * 0.85, self.y + s * 0.85, fillColor=self.crossColor, strokeColor=self.crossColor, strokeWidth=self.crosswidth)
        g.add(crossLine1)
        crossLine2 = shapes.Line(self.x + s * 0.15, self.y + s * 0.85, self.x + s * 0.85, self.y + s * 0.15, fillColor=self.crossColor, strokeColor=self.crossColor, strokeWidth=self.crosswidth)
        g.add(crossLine2)
        return g