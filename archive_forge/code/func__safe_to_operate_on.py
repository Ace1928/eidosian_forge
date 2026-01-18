import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def _safe_to_operate_on(self):
    """Return whether it is safe to interact with this context.

        This is considered to be the case if it's the current context and this
        method is called from the main thread.
        """
    return gl.current_context is self and threading.current_thread() is threading.main_thread()