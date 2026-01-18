from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _set_x(self, x):
    if x == self.bbox[0][0]:
        return False
    self.pos = (x, self.y)
    return True