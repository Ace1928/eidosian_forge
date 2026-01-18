from kivy.logger import Logger
from kivy.core.window import WindowBase
from kivy.base import EventLoop, ExceptionManager, stopTouchApp
from kivy.lib.vidcore_lite import bcm, egl
from os import environ
def _create_egl_context(self, win, flags):
    api = egl._constants.EGL_OPENGL_ES_API
    c = egl._constants
    attribs = [c.EGL_RED_SIZE, 8, c.EGL_GREEN_SIZE, 8, c.EGL_BLUE_SIZE, 8, c.EGL_ALPHA_SIZE, 8, c.EGL_DEPTH_SIZE, 16, c.EGL_STENCIL_SIZE, 8, c.EGL_SURFACE_TYPE, c.EGL_WINDOW_BIT, c.EGL_NONE]
    attribs_context = [c.EGL_CONTEXT_CLIENT_VERSION, 2, c.EGL_NONE]
    display = egl.GetDisplay(c.EGL_DEFAULT_DISPLAY)
    egl.Initialise(display)
    egl.BindAPI(c.EGL_OPENGL_ES_API)
    egl.GetConfigs(display)
    config = egl.ChooseConfig(display, attribs, 1)[0]
    surface = egl.CreateWindowSurface(display, config, win)
    context = egl.CreateContext(display, config, None, attribs_context)
    egl.MakeCurrent(display, surface, surface, context)
    self.egl_info = (display, surface, context)
    egl.MakeCurrent(display, surface, surface, context)