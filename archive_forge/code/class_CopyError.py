import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class CopyError(Error):
    """Raised to indicate that operation requires data-copying
    
    if you set:
        OpenGL.ERROR_ON_COPY = True 
    
    before importing OpenGL.GL, this error will be raised when 
    a passed argument would require a copy to be made.
    """