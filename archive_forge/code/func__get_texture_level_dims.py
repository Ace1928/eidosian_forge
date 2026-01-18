from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
def _get_texture_level_dims(target, level):
    """Retrieve texture dims for given level and target"""
    dims = []
    dim = _types.GLuint()
    GL_1_1.glGetTexLevelParameteriv(target, level, GL_1_1.GL_TEXTURE_WIDTH, dim)
    dims = [dim.value]
    if target != GL_1_1.GL_TEXTURE_1D:
        GL_1_1.glGetTexLevelParameteriv(target, level, GL_1_1.GL_TEXTURE_HEIGHT, dim)
        dims.append(dim.value)
        if target != GL_1_1.GL_TEXTURE_2D:
            GL_1_1.glGetTexLevelParameteriv(target, level, GL_1_2.GL_TEXTURE_DEPTH, dim)
            dims.append(dim.value)
    return dims