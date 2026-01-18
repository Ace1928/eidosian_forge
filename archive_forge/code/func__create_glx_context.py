import warnings
from ctypes import *
from .base import Config, CanvasConfig, Context
from pyglet.canvas.xlib import XlibCanvas
from pyglet.gl import glx
from pyglet.gl import glxext_arb
from pyglet.gl import glx_info
from pyglet.gl import glxext_mesa
from pyglet.gl import lib
from pyglet import gl
def _create_glx_context(self, share):
    if share:
        share_context = share.glx_context
    else:
        share_context = None
    attribs = []
    if self.config.major_version is not None:
        attribs.extend([glxext_arb.GLX_CONTEXT_MAJOR_VERSION_ARB, self.config.major_version])
    if self.config.minor_version is not None:
        attribs.extend([glxext_arb.GLX_CONTEXT_MINOR_VERSION_ARB, self.config.minor_version])
    if self.config.opengl_api == 'gl':
        attribs.extend([glxext_arb.GLX_CONTEXT_PROFILE_MASK_ARB, glxext_arb.GLX_CONTEXT_CORE_PROFILE_BIT_ARB])
    elif self.config.opengl_api == 'gles':
        attribs.extend([glxext_arb.GLX_CONTEXT_PROFILE_MASK_ARB, glxext_arb.GLX_CONTEXT_ES2_PROFILE_BIT_EXT])
    flags = 0
    if self.config.forward_compatible:
        flags |= glxext_arb.GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB
    if self.config.debug:
        flags |= glxext_arb.GLX_CONTEXT_DEBUG_BIT_ARB
    if flags:
        attribs.extend([glxext_arb.GLX_CONTEXT_FLAGS_ARB, flags])
    attribs.append(0)
    attribs = (c_int * len(attribs))(*attribs)
    return glxext_arb.glXCreateContextAttribsARB(self.config.canvas.display._display, self.config.fbconfig, share_context, True, attribs)