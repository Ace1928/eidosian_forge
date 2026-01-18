from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class ArrowOne(_Symbol):
    """This widget draws an arrow (style one).

        possible attributes:
        'x', 'y', 'size', 'fillColor'

        """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.fillColor = colors.red
        self.strokeWidth = 0
        self.strokeColor = None

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        x = self.x
        y = self.y
        s2 = s / 2
        s3 = s / 3
        s5 = s / 5
        g.add(shapes.Polygon(points=[x, y + s3, x, y + 2 * s3, x + s2, y + 2 * s3, x + s2, y + 4 * s5, x + s, y + s2, x + s2, y + s5, x + s2, y + s3], fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=self.strokeWidth))
        return g