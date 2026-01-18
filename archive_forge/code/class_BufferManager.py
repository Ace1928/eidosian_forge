import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
class BufferManager:
    """Manages the set of framebuffers for a context.

    Use :py:func:`~pyglet.image.get_buffer_manager` to obtain the instance of this class for the
    current context.
    """

    def __init__(self):
        self._color_buffer = None
        self._depth_buffer = None
        self.free_stencil_bits = None
        self._refs = []

    @staticmethod
    def get_viewport():
        """Get the current OpenGL viewport dimensions.

        :rtype: 4-tuple of float.
        :return: Left, top, right and bottom dimensions.
        """
        viewport = (GLint * 4)()
        glGetIntegerv(GL_VIEWPORT, viewport)
        return viewport

    def get_color_buffer(self):
        """Get the color buffer.

        :rtype: :py:class:`~pyglet.image.ColorBufferImage`
        """
        viewport = self.get_viewport()
        viewport_width = viewport[2]
        viewport_height = viewport[3]
        if not self._color_buffer or viewport_width != self._color_buffer.width or viewport_height != self._color_buffer.height:
            self._color_buffer = ColorBufferImage(*viewport)
        return self._color_buffer

    def get_depth_buffer(self):
        """Get the depth buffer.

        :rtype: :py:class:`~pyglet.image.DepthBufferImage`
        """
        viewport = self.get_viewport()
        viewport_width = viewport[2]
        viewport_height = viewport[3]
        if not self._depth_buffer or viewport_width != self._depth_buffer.width or viewport_height != self._depth_buffer.height:
            self._depth_buffer = DepthBufferImage(*viewport)
        return self._depth_buffer

    def get_buffer_mask(self):
        """Get a free bitmask buffer.

        A bitmask buffer is a buffer referencing a single bit in the stencil
        buffer.  If no bits are free, `ImageException` is raised.  Bits are
        released when the bitmask buffer is garbage collected.

        :rtype: :py:class:`~pyglet.image.BufferImageMask`
        """
        if self.free_stencil_bits is None:
            try:
                stencil_bits = GLint()
                glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_STENCIL, GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE, stencil_bits)
                self.free_stencil_bits = list(range(stencil_bits.value))
            except GLException:
                pass
        if not self.free_stencil_bits:
            raise ImageException('No free stencil bits are available.')
        stencil_bit = self.free_stencil_bits.pop(0)
        x, y, width, height = self.get_viewport()
        bufimg = BufferImageMask(x, y, width, height)
        bufimg.stencil_bit = stencil_bit

        def release_buffer(ref, owner=self):
            owner.free_stencil_bits.insert(0, stencil_bit)
        self._refs.append(weakref.ref(bufimg, release_buffer))
        return bufimg