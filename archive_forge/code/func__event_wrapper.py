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
def _event_wrapper(f):
    f._platform_event = True
    if not hasattr(f, '_platform_event_data'):
        f._platform_event_data = []
    f._platform_event_data.append(data)
    return f