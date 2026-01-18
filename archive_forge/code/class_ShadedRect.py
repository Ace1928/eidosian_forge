from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
class ShadedRect(Widget):
    """This makes a rectangle with shaded colors between two colors.

    Colors are interpolated linearly between 'fillColorStart'
    and 'fillColorEnd', both of which appear at the margins.
    If 'numShades' is set to one, though, only 'fillColorStart'
    is used.
    """
    _attrMap = AttrMap(x=AttrMapValue(isNumber, desc="The grid's lower-left x position."), y=AttrMapValue(isNumber, desc="The grid's lower-left y position."), width=AttrMapValue(isNumber, desc="The grid's width."), height=AttrMapValue(isNumber, desc="The grid's height."), orientation=AttrMapValue(OneOf(('vertical', 'horizontal')), desc='Determines if stripes are vertical or horizontal.'), numShades=AttrMapValue(isNumber, desc='The number of interpolating colors.'), fillColorStart=AttrMapValue(isColorOrNone, desc='Start value of the color shade.'), fillColorEnd=AttrMapValue(isColorOrNone, desc='End value of the color shade.'), strokeColor=AttrMapValue(isColorOrNone, desc='Color used for border line.'), strokeWidth=AttrMapValue(isNumber, desc='Width used for lines.'), cylinderMode=AttrMapValue(isBoolean, desc='True if shading reverses in middle.'))

    def __init__(self, **kw):
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        self.orientation = 'vertical'
        self.numShades = 20
        self.fillColorStart = colors.pink
        self.fillColorEnd = colors.black
        self.strokeColor = colors.black
        self.strokeWidth = 2
        self.cylinderMode = 0
        self.setProperties(kw)

    def demo(self):
        D = Drawing(100, 100)
        g = ShadedRect()
        D.add(g)
        return D

    def _flipRectCorners(self):
        """Flip rectangle's corners if width or height is negative."""
        x, y, width, height, fillColorStart, fillColorEnd = (self.x, self.y, self.width, self.height, self.fillColorStart, self.fillColorEnd)
        if width < 0 and height > 0:
            x = x + width
            width = -width
            if self.orientation == 'vertical':
                fillColorStart, fillColorEnd = (fillColorEnd, fillColorStart)
        elif height < 0 and width > 0:
            y = y + height
            height = -height
            if self.orientation == 'horizontal':
                fillColorStart, fillColorEnd = (fillColorEnd, fillColorStart)
        elif height < 0 and height < 0:
            x = x + width
            width = -width
            y = y + height
            height = -height
        return (x, y, width, height, fillColorStart, fillColorEnd)

    def draw(self):
        group = Group()
        x, y, w, h, c0, c1 = self._flipRectCorners()
        numShades = self.numShades
        if self.cylinderMode:
            if not numShades % 2:
                numShades = numShades + 1
            halfNumShades = int((numShades - 1) / 2) + 1
        num = float(numShades)
        vertical = self.orientation == 'vertical'
        if vertical:
            if numShades == 1:
                V = [x]
            else:
                V = frange(x, x + w, w / num)
        elif numShades == 1:
            V = [y]
        else:
            V = frange(y, y + h, h / num)
        for v in V:
            stripe = vertical and Rect(v, y, w / num, h) or Rect(x, v, w, h / num)
            if self.cylinderMode:
                if V.index(v) >= halfNumShades:
                    col = colors.linearlyInterpolatedColor(c1, c0, V[halfNumShades], V[-1], v)
                else:
                    col = colors.linearlyInterpolatedColor(c0, c1, V[0], V[halfNumShades], v)
            else:
                col = colors.linearlyInterpolatedColor(c0, c1, V[0], V[-1], v)
            stripe.fillColor = col
            stripe.strokeColor = col
            stripe.strokeWidth = 1
            group.add(stripe)
        if self.strokeColor and self.strokeWidth >= 0:
            rect = Rect(x, y, w, h)
            rect.strokeColor = self.strokeColor
            rect.strokeWidth = self.strokeWidth
            rect.fillColor = None
            group.add(rect)
        return group