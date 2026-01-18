from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class ScaleWidget(Widget):
    """Contents with a scale and offset"""
    _attrMap = AttrMap(x=AttrMapValue(isNumber, desc='x offset'), y=AttrMapValue(isNumber, desc='y offset'), scale=AttrMapValue(isNumber, desc='scale'), contents=AttrMapValue(None, desc='Contained drawable elements'))

    def __init__(self, x=0, y=0, scale=1.0, contents=None):
        self.x = x
        self.y = y
        if not contents:
            contents = []
        elif not isinstance(contents, (tuple, list)):
            contents = (contents,)
        self.contents = list(contents)
        self.scale = scale

    def draw(self):
        return shapes.Group(*self.contents, transform=(self.scale, 0, 0, self.scale, self.x, self.y))