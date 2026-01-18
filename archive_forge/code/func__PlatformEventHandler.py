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
def _PlatformEventHandler(data):
    """Decorator for platform event handlers.

    Apply giving the platform-specific data needed by the window to associate
    the method with an event.  See platform-specific subclasses of this
    decorator for examples.

    The following attributes are set on the function, which is returned
    otherwise unchanged:

    _platform_event
        True
    _platform_event_data
        List of data applied to the function (permitting multiple decorators
        on the same method).
    """

    def _event_wrapper(f):
        f._platform_event = True
        if not hasattr(f, '_platform_event_data'):
            f._platform_event_data = []
        f._platform_event_data.append(data)
        return f
    return _event_wrapper