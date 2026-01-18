from pyglet.canvas.win32 import Win32Canvas
from .base import Config, CanvasConfig, Context
from pyglet import gl
from pyglet.gl import gl_info
from pyglet.gl import wgl
from pyglet.gl import wglext_arb
from pyglet.gl import wgl_info
from pyglet.libs.win32 import _user32, _kernel32, _gdi32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class Win32ARBContext(_BaseWin32Context):

    def attach(self, canvas):
        share = self.context_share
        if share:
            if not share.canvas:
                raise RuntimeError('Share context has no canvas.')
            share = share._context
        attribs = []
        if self.config.major_version is not None:
            attribs.extend([wglext_arb.WGL_CONTEXT_MAJOR_VERSION_ARB, self.config.major_version])
        if self.config.minor_version is not None:
            attribs.extend([wglext_arb.WGL_CONTEXT_MINOR_VERSION_ARB, self.config.minor_version])
        flags = 0
        if self.config.forward_compatible:
            flags |= wglext_arb.WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB
        if self.config.debug:
            flags |= wglext_arb.WGL_CONTEXT_DEBUG_BIT_ARB
        if flags:
            attribs.extend([wglext_arb.WGL_CONTEXT_FLAGS_ARB, flags])
        attribs.append(0)
        attribs = (c_int * len(attribs))(*attribs)
        self.config._set_pixel_format(canvas)
        self._context = wglext_arb.wglCreateContextAttribsARB(canvas.hdc, share, attribs)
        super().attach(canvas)