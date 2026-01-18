import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
class CanvasConfig(Config):
    """An OpenGL configuration for a particular canvas.

    Use `Config.match` to obtain an instance of this class.

    .. versionadded:: 1.2

    :Ivariables:
        `canvas` : `Canvas`
            The canvas this config is valid on.

    """

    def __init__(self, canvas, base_config):
        self.canvas = canvas
        self.major_version = base_config.major_version
        self.minor_version = base_config.minor_version
        self.forward_compatible = base_config.forward_compatible
        self.opengl_api = base_config.opengl_api or self.opengl_api
        self.debug = base_config.debug

    def compatible(self, canvas):
        raise NotImplementedError('abstract')

    def create_context(self, share):
        """Create a GL context that satisifies this configuration.

        :Parameters:
            `share` : `Context`
                If not None, a context with which to share objects with.

        :rtype: `Context`
        :return: The new context.
        """
        raise NotImplementedError('abstract')

    def is_complete(self):
        return True