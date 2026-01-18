import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
@property
def _min_y(self):
    return self._y - self._half_knob_height