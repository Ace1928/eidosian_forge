from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class TwoFaces(Widget):

    def __init__(self):
        self.faceOne = Face()
        self.faceOne.mood = 'happy'
        self.faceTwo = Face()
        self.faceTwo.x = 100
        self.faceTwo.mood = 'sad'

    def draw(self):
        """Just return a group"""
        return shapes.Group(self.faceOne, self.faceTwo)

    def demo(self):
        """The default case already looks good enough,
        no implementation needed here"""
        pass