import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class NoContext(Error):
    """Raised to indicate that there is no currently active context
    
    Technically almost *any* OpenGL call can segfault if there is 
    no active context.  The OpenGL.CHECK_CONTEXT flag, if enabled 
    will cause this error to be raised whenever a GL or GLU call is 
    issued (via PyOpenGL) if there is no currently valid context.
    """