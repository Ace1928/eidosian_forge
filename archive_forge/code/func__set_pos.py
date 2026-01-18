from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _set_pos(self, pos):
    _pos = self.bbox[0]
    if pos == _pos:
        return
    t = Vector(*pos) - _pos
    trans = Matrix().translate(t.x, t.y, 0)
    self.apply_transform(trans)