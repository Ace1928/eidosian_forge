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
class _BaseWin32Context(Context):

    def __init__(self, config, share):
        super().__init__(config, share)
        self._context = None

    def set_current(self):
        if self._context is not None and self != gl.current_context:
            wgl.wglMakeCurrent(self.canvas.hdc, self._context)
        super().set_current()

    def detach(self):
        if self.canvas:
            wgl.wglDeleteContext(self._context)
            self._context = None
        super().detach()

    def flip(self):
        _gdi32.SwapBuffers(self.canvas.hdc)

    def get_vsync(self):
        if wgl_info.have_extension('WGL_EXT_swap_control'):
            return bool(wglext_arb.wglGetSwapIntervalEXT())

    def set_vsync(self, vsync):
        if wgl_info.have_extension('WGL_EXT_swap_control'):
            wglext_arb.wglSwapIntervalEXT(int(vsync))