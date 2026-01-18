from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class _Symbol(Widget):
    """Abstract base widget
    possible attributes:
    'x', 'y', 'size', 'fillColor', 'strokeColor'
    """
    _nodoc = 1
    _attrMap = AttrMap(x=AttrMapValue(isNumber, desc='symbol x coordinate'), y=AttrMapValue(isNumber, desc='symbol y coordinate'), dx=AttrMapValue(isNumber, desc='symbol x coordinate adjustment'), dy=AttrMapValue(isNumber, desc='symbol x coordinate adjustment'), size=AttrMapValue(isNumber), fillColor=AttrMapValue(isColorOrNone), strokeColor=AttrMapValue(isColorOrNone), strokeWidth=AttrMapValue(isNumber))

    def __init__(self):
        assert self.__class__.__name__ != '_Symbol', 'Abstract class _Symbol instantiated'
        self.x = self.y = self.dx = self.dy = 0
        self.size = 100
        self.fillColor = colors.red
        self.strokeColor = None
        self.strokeWidth = 0.1

    def demo(self):
        D = shapes.Drawing(200, 100)
        s = float(self.size)
        ob = self.__class__()
        ob.x = 50
        ob.y = 0
        ob.draw()
        D.add(ob)
        D.add(shapes.String(ob.x + s / 2, ob.y - 12, ob.__class__.__name__, fillColor=colors.black, textAnchor='middle', fontSize=10))
        return D