import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class GLUError(Error):
    """GLU error implementation class"""