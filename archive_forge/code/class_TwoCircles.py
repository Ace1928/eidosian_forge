from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class TwoCircles(Widget):

    def __init__(self):
        self.leftCircle = shapes.Circle(100, 100, 20, fillColor=colors.red)
        self.rightCircle = shapes.Circle(300, 100, 20, fillColor=colors.red)

    def draw(self):
        return shapes.Group(self.leftCircle, self.rightCircle)