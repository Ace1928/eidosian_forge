from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def _update_canvas_bbox(self, g):
    if not hasattr(g, '_bbrect'):
        return
    bb = g.bbox
    g._bbrect.pos = (bb['minx'], bb['miny'])
    g._bbrect.size = (bb['maxx'] - bb['minx'], bb['maxy'] - bb['miny'])