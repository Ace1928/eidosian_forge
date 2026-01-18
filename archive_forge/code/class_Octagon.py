from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class Octagon(_Symbol):
    """This widget draws an Octagon.

        possible attributes:
        'x', 'y', 'size', 'fillColor', 'strokeColor'

    """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.fillColor = colors.yellow
        self.strokeColor = None

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        athird = s / 3
        octagon = shapes.Polygon(points=[self.x + athird, self.y, self.x, self.y + athird, self.x, self.y + athird * 2, self.x + athird, self.y + s, self.x + athird * 2, self.y + s, self.x + s, self.y + athird * 2, self.x + s, self.y + athird, self.x + athird * 2, self.y], strokeColor=self.strokeColor, fillColor=self.fillColor, strokeWidth=10)
        g.add(octagon)
        return g