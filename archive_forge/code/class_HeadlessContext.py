from ctypes import *
from pyglet import gl
from pyglet.canvas.headless import HeadlessCanvas
from pyglet.libs.egl import egl
from pyglet.libs.egl.egl import *
from .base import CanvasConfig, Config, Context
class HeadlessContext(Context):

    def __init__(self, config, share):
        super(HeadlessContext, self).__init__(config, share)
        self.display_connection = config.canvas.display._display_connection
        self.egl_context = self._create_egl_context(share)
        if not self.egl_context:
            raise gl.ContextException('Could not create GL context')

    def _create_egl_context(self, share):
        if share:
            share_context = share.egl_context
        else:
            share_context = None
        if self.config.opengl_api == 'gl':
            egl.eglBindAPI(egl.EGL_OPENGL_API)
        elif self.config.opengl_api == 'gles':
            egl.eglBindAPI(egl.EGL_OPENGL_ES_API)
        return egl.eglCreateContext(self.config.canvas.display._display_connection, self.config._egl_config, share_context, self.config._context_attrib_array)

    def attach(self, canvas):
        if canvas is self.canvas:
            return
        super(HeadlessContext, self).attach(canvas)
        self.egl_surface = canvas.egl_surface
        self.set_current()

    def set_current(self):
        egl.eglMakeCurrent(self.display_connection, self.egl_surface, self.egl_surface, self.egl_context)
        super(HeadlessContext, self).set_current()

    def detach(self):
        if not self.canvas:
            return
        self.set_current()
        gl.glFlush()
        super(HeadlessContext, self).detach()
        egl.eglMakeCurrent(self.display_connection, 0, 0, None)
        self.egl_surface = None

    def destroy(self):
        super(HeadlessContext, self).destroy()
        if self.egl_context:
            egl.eglDestroyContext(self.display_connection, self.egl_context)
            self.egl_context = None

    def flip(self):
        if not self.egl_surface:
            return
        egl.eglSwapBuffers(self.display_connection, self.egl_surface)