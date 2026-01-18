from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _set_y(self, y):
    if y == self.bbox[0][1]:
        return False
    self.pos = (self.x, y)
    return True