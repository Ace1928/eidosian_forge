from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def accept_stroke(self, count=1):
    """Returns True if this container can accept `count` new strokes"""
    if not self.max_strokes:
        return True
    return len(self._strokes) + count <= self.max_strokes