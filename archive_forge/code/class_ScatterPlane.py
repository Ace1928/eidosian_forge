from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
class ScatterPlane(Scatter):
    """This is essentially an unbounded Scatter widget. It's a convenience
       class to make it easier to handle infinite planes.
    """

    def __init__(self, **kwargs):
        if 'auto_bring_to_front' not in kwargs:
            self.auto_bring_to_front = False
        super(ScatterPlane, self).__init__(**kwargs)

    def collide_point(self, x, y):
        return True