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
class NoSuchScreenModeException(WindowException):
    """An exception indicating the requested screen resolution could not be
    met."""
    pass