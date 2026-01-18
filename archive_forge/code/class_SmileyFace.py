from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class SmileyFace(_Symbol):
    """This draws a classic smiley face.

        possible attributes:
        'x', 'y', 'size', 'fillColor'

    """

    def __init__(self):
        _Symbol.__init__(self)
        self.x = 0
        self.y = 0
        self.size = 100
        self.fillColor = colors.yellow
        self.strokeColor = colors.black

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        g.add(shapes.Circle(cx=self.x + s / 2, cy=self.y + s / 2, r=s / 2, fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=max(s / 38.0, self.strokeWidth)))
        for i in (1, 2):
            g.add(shapes.Ellipse(self.x + s / 3 * i, self.y + s / 3 * 2, s / 30, s / 10, fillColor=self.strokeColor, strokeColor=self.strokeColor, strokeWidth=max(s / 38.0, self.strokeWidth)))
        centerx = self.x + s / 2
        centery = self.y + s / 2
        radius = s / 3
        yradius = radius
        xradius = radius
        startangledegrees = 200
        endangledegrees = 340
        degreedelta = 1
        pointslist = []
        a = pointslist.append
        from math import sin, cos, pi
        degreestoradians = pi / 180.0
        radiansdelta = degreedelta * degreestoradians
        startangle = startangledegrees * degreestoradians
        endangle = endangledegrees * degreestoradians
        while endangle < startangle:
            endangle = endangle + 2 * pi
        angle = startangle
        while angle < endangle:
            x = centerx + cos(angle) * radius
            y = centery + sin(angle) * yradius
            a(x)
            a(y)
            angle = angle + radiansdelta
        smile = shapes.PolyLine(pointslist, fillColor=self.strokeColor, strokeColor=self.strokeColor, strokeWidth=max(s / 38.0, self.strokeWidth))
        g.add(smile)
        return g