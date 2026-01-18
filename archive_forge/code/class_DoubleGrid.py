from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
class DoubleGrid(Widget):
    """This combines two ordinary Grid objects orthogonal to each other.
    """
    _attrMap = AttrMap(x=AttrMapValue(isNumber, desc="The grid's lower-left x position."), y=AttrMapValue(isNumber, desc="The grid's lower-left y position."), width=AttrMapValue(isNumber, desc="The grid's width."), height=AttrMapValue(isNumber, desc="The grid's height."), grid0=AttrMapValue(None, desc='The first grid component.'), grid1=AttrMapValue(None, desc='The second grid component.'))

    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        g0 = Grid()
        g0.x = self.x
        g0.y = self.y
        g0.width = self.width
        g0.height = self.height
        g0.orientation = 'vertical'
        g0.useLines = 1
        g0.useRects = 0
        g0.delta = 20
        g0.delta0 = 0
        g0.deltaSteps = []
        g0.fillColor = colors.white
        g0.stripeColors = [colors.red, colors.green, colors.blue]
        g0.strokeColor = colors.black
        g0.strokeWidth = 1
        g1 = Grid()
        g1.x = self.x
        g1.y = self.y
        g1.width = self.width
        g1.height = self.height
        g1.orientation = 'horizontal'
        g1.useLines = 1
        g1.useRects = 0
        g1.delta = 20
        g1.delta0 = 0
        g1.deltaSteps = []
        g1.fillColor = colors.white
        g1.stripeColors = [colors.red, colors.green, colors.blue]
        g1.strokeColor = colors.black
        g1.strokeWidth = 1
        self.grid0 = g0
        self.grid1 = g1

    def demo(self):
        D = Drawing(100, 100)
        g = DoubleGrid()
        D.add(g)
        return D

    def draw(self):
        group = Group()
        g0, g1 = (self.grid0, self.grid1)
        G = g0.useRects == 1 and g1.useRects == 0 and (g0, g1) or (g1, g0)
        for g in G:
            group.add(g.makeOuterRect())
        for g in G:
            group.add(g.makeInnerTiles())
            group.add(g.makeInnerLines(), name='_gridLines')
        return group