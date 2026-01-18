import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def _update_position(self):
    self._layout.position = (self._x, self._y, 0)
    self._outline.position = (self._x - self._pad, self._y - self._pad)