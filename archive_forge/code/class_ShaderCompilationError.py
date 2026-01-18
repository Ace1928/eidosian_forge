import logging
from OpenGL import GL
from OpenGL.GL.ARB import (
from OpenGL.extensions import alternate
from OpenGL._bytes import bytes,unicode,as_8_bit
class ShaderCompilationError(RuntimeError):
    """Raised when a shader compilation fails"""