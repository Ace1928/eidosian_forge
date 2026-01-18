from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class _GLXQuerier(extensions.ExtensionQuerier):
    prefix = as_8_bit('GLX_')
    assumed_version = [1, 1]
    version_prefix = as_8_bit('GLX_VERSION_GLX_')

    def getDisplay(self):
        from OpenGL.raw.GLX import _types
        from OpenGL.platform import ctypesloader
        import ctypes, os
        X11 = ctypesloader.loadLibrary(ctypes.cdll, 'X11')
        XOpenDisplay = X11.XOpenDisplay
        XOpenDisplay.restype = ctypes.POINTER(_types.Display)
        return XOpenDisplay(os.environ.get('DISPLAY'))

    def getScreen(self, display):
        from OpenGL.platform import ctypesloader
        from OpenGL.raw.GLX import _types
        import ctypes, os
        X11 = ctypesloader.loadLibrary(ctypes.cdll, 'X11')
        XDefaultScreen = X11.XDefaultScreen
        XDefaultScreen.argtypes = [ctypes.POINTER(_types.Display)]
        return XDefaultScreen(display)

    def pullVersion(self):
        from OpenGL.GLX import glXQueryVersion
        import ctypes
        if glXQueryVersion:
            display = self.getDisplay()
            major, minor = (ctypes.c_int(), ctypes.c_int())
            glXQueryVersion(display, major, minor)
            return [major.value, minor.value]
        else:
            return [1, 1]

    def pullExtensions(self):
        if self.getVersion() >= [1, 2]:
            from OpenGL.GLX import glXQueryExtensionsString
            display = self.getDisplay()
            screen = self.getScreen(display)
            if glXQueryExtensionsString:
                return glXQueryExtensionsString(display, screen).split()
        return []