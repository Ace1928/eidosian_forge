import sys
from typing import Tuple
import pyglet
import pyglet.window.key
import pyglet.window.mouse
from pyglet import gl
from pyglet.math import Mat4
from pyglet.event import EventDispatcher
from pyglet.window import key, event
from pyglet.graphics import shader
class MouseCursor:
    """An abstract mouse cursor."""
    gl_drawable = True
    hw_drawable = False

    def draw(self, x, y):
        """Abstract render method.

        The cursor should be drawn with the "hot" spot at the given
        coordinates.  The projection is set to the pyglet default (i.e.,
        orthographic in window-space), however no other aspects of the
        state can be assumed.

        :Parameters:
            `x` : int
                X coordinate of the mouse pointer's hot spot.
            `y` : int
                Y coordinate of the mouse pointer's hot spot.

        """
        pass