from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
class GLXInfo:

    def __init__(self, display=None):
        if display and (not _glx_info.display):
            _glx_info.set_display(display)
        self.display = display

    def set_display(self, display):
        self.display = display

    def check_display(self):
        if not self.display:
            raise GLXInfoException('No X11 display has been set yet.')

    def have_version(self, major, minor=0):
        self.check_display()
        if not glXQueryExtension(self.display, None, None):
            raise GLXInfoException('pyglet requires an X server with GLX')
        server_version = self.get_server_version().split()[0]
        client_version = self.get_client_version().split()[0]
        server = [int(i) for i in server_version.split('.')]
        client = [int(i) for i in client_version.split('.')]
        return tuple(server) >= (major, minor) and tuple(client) >= (major, minor)

    def get_server_vendor(self):
        self.check_display()
        return asstr(glXQueryServerString(self.display, 0, GLX_VENDOR))

    def get_server_version(self):
        self.check_display()
        major = c_int()
        minor = c_int()
        if not glXQueryVersion(self.display, byref(major), byref(minor)):
            raise GLXInfoException('Could not determine GLX server version')
        return f'{major.value}.{minor.value}'

    def get_server_extensions(self):
        self.check_display()
        return asstr(glXQueryServerString(self.display, 0, GLX_EXTENSIONS)).split()

    def get_client_vendor(self):
        self.check_display()
        return asstr(glXGetClientString(self.display, GLX_VENDOR))

    def get_client_version(self):
        self.check_display()
        return asstr(glXGetClientString(self.display, GLX_VERSION))

    def get_client_extensions(self):
        self.check_display()
        return asstr(glXGetClientString(self.display, GLX_EXTENSIONS)).split()

    def get_extensions(self):
        self.check_display()
        return asstr(glXQueryExtensionsString(self.display, 0)).split()

    def have_extension(self, extension):
        self.check_display()
        if not self.have_version(1, 1):
            return False
        return extension in self.get_extensions()