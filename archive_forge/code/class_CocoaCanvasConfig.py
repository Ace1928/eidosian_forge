import platform
from ctypes import c_uint32, c_int, byref
from pyglet.gl.base import Config, CanvasConfig, Context
from pyglet.gl import ContextException
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, quartz
class CocoaCanvasConfig(CanvasConfig):

    def __init__(self, canvas, config, pixel_format):
        super(CocoaCanvasConfig, self).__init__(canvas, config)
        self._pixel_format = pixel_format
        for name, attr in _gl_attributes.items():
            vals = c_int()
            self._pixel_format.getValues_forAttribute_forVirtualScreen_(byref(vals), attr, 0)
            setattr(self, name, vals.value)
        for name, value in _fake_gl_attributes.items():
            setattr(self, name, value)
        if _os_x_version >= os_x_release['lion']:
            vals = c_int()
            profile = self._pixel_format.getValues_forAttribute_forVirtualScreen_(byref(vals), cocoapy.NSOpenGLPFAOpenGLProfile, 0)
            if vals.value == cocoapy.NSOpenGLProfileVersion4_1Core:
                setattr(self, 'major_version', 4)
                setattr(self, 'minor_version', 1)
            elif vals.value == cocoapy.NSOpenGLProfileVersion3_2Core:
                setattr(self, 'major_version', 3)
                setattr(self, 'minor_version', 2)
            else:
                setattr(self, 'major_version', 2)
                setattr(self, 'minor_version', 1)

    def create_context(self, share):
        if share:
            share_context = share._nscontext
        else:
            share_context = None
        nscontext = NSOpenGLContext.alloc().initWithFormat_shareContext_(self._pixel_format, share_context)
        return CocoaContext(self, nscontext, share)

    def compatible(self, canvas):
        return isinstance(canvas, CocoaCanvas)