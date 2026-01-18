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
class _WindowMetaclass(type):
    """Sets the _platform_event_names class variable on the window
    subclass.
    """

    def __init__(cls, name, bases, dict):
        cls._platform_event_names = set()
        for base in bases:
            if hasattr(base, '_platform_event_names'):
                cls._platform_event_names.update(base._platform_event_names)
        for name, func in dict.items():
            if hasattr(func, '_platform_event'):
                cls._platform_event_names.add(name)
        super(_WindowMetaclass, cls).__init__(name, bases, dict)