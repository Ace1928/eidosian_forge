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
class ImageMouseCursor(MouseCursor):
    """A user-defined mouse cursor created from an image.

    Use this class to create your own mouse cursors and assign them
    to windows. Cursors can be drawn by OpenGL, or optionally passed
    to the OS to render natively. There are no restrictions on cursors
    drawn by OpenGL, but natively rendered cursors may have some
    platform limitations (such as color depth, or size). In general,
    reasonably sized cursors will render correctly
    """

    def __init__(self, image, hot_x=0, hot_y=0, acceleration=False):
        """Create a mouse cursor from an image.

        :Parameters:
            `image` : `pyglet.image.AbstractImage`
                Image to use for the mouse cursor.  It must have a
                valid ``texture`` attribute.
            `hot_x` : int
                X coordinate of the "hot" spot in the image relative to the
                image's anchor. May be clamped to the maximum image width
                if ``acceleration=True``.
            `hot_y` : int
                Y coordinate of the "hot" spot in the image, relative to the
                image's anchor. May be clamped to the maximum image height
                if ``acceleration=True``.
            `acceleration` : int
                If True, draw the cursor natively instead of usign OpenGL.
                The image may be downsampled or color reduced to fit the
                platform limitations.
        """
        self.texture = image.get_texture()
        self.hot_x = hot_x
        self.hot_y = hot_y
        self.gl_drawable = not acceleration
        self.hw_drawable = acceleration

    def draw(self, x, y):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.texture.blit(x - self.hot_x, y - self.hot_y, 0)
        gl.glDisable(gl.GL_BLEND)