import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class NullFunctionError(Error):
    """Error raised when an undefined function is called"""