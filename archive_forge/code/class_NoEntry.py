from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class NoEntry(_Symbol):
    """This draws a (British) No Entry sign - a red circle with a white line on it.

        possible attributes:
        'x', 'y', 'size'

        """
    _attrMap = AttrMap(BASE=_Symbol, innerBarColor=AttrMapValue(isColorOrNone, desc='color of the inner bar'))

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.strokeColor = colors.black
        self.fillColor = colors.orangered
        self.innerBarColor = colors.ghostwhite

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        if self.strokeColor:
            g.add(shapes.Circle(cx=self.x + s / 2, cy=self.y + s / 2, r=s / 2, fillColor=None, strokeColor=self.strokeColor, strokeWidth=1))
        if self.fillColor:
            g.add(shapes.Circle(cx=self.x + s / 2, cy=self.y + s / 2, r=s / 2 - s / 50, fillColor=self.fillColor, strokeColor=None, strokeWidth=0))
        innerBarColor = self.innerBarColor
        if innerBarColor:
            g.add(shapes.Rect(self.x + s * 0.1, self.y + s * 0.4, width=s * 0.8, height=s * 0.2, fillColor=innerBarColor, strokeColor=innerBarColor, strokeLineCap=1, strokeWidth=0))
        return g