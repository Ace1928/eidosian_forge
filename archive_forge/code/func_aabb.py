import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
@property
def aabb(self):
    """Bounding box of the widget.

        Expressed as (x, y, x + width, y + height)

        :type: (int, int, int, int)
        """
    return (self._x, self._y, self._x + self._width, self._y + self._height)