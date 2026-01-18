from ctypes import *
from pyglet import gl
from pyglet.canvas.headless import HeadlessCanvas
from pyglet.libs.egl import egl
from pyglet.libs.egl.egl import *
from .base import CanvasConfig, Config, Context
class HeadlessCanvasConfig(CanvasConfig):
    attribute_ids = {'buffer_size': egl.EGL_BUFFER_SIZE, 'level': egl.EGL_LEVEL, 'red_size': egl.EGL_RED_SIZE, 'green_size': egl.EGL_GREEN_SIZE, 'blue_size': egl.EGL_BLUE_SIZE, 'alpha_size': egl.EGL_ALPHA_SIZE, 'depth_size': egl.EGL_DEPTH_SIZE, 'stencil_size': egl.EGL_STENCIL_SIZE, 'sample_buffers': egl.EGL_SAMPLE_BUFFERS, 'samples': egl.EGL_SAMPLES}

    def __init__(self, canvas, egl_config, config):
        super(HeadlessCanvasConfig, self).__init__(canvas, config)
        self._egl_config = egl_config
        context_attribs = (EGL_CONTEXT_MAJOR_VERSION, config.major_version or 2, EGL_CONTEXT_MINOR_VERSION, config.minor_version or 0, EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE, config.forward_compatible or 0, EGL_CONTEXT_OPENGL_DEBUG, config.debug or 0, EGL_NONE)
        self._context_attrib_array = (egl.EGLint * len(context_attribs))(*context_attribs)
        for name, attr in self.attribute_ids.items():
            value = egl.EGLint()
            egl.eglGetConfigAttrib(canvas.display._display_connection, egl_config, attr, byref(value))
            setattr(self, name, value.value)
        for name, value in _fake_gl_attributes.items():
            setattr(self, name, value)

    def compatible(self, canvas):
        return isinstance(canvas, HeadlessCanvas)

    def create_context(self, share):
        return HeadlessContext(self, share)