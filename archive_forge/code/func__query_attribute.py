import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _query_attribute(program_id: int, index: int):
    """Query the name, type, and size of an Attribute by index."""
    asize = GLint()
    atype = GLenum()
    buf_size = 192
    aname = create_string_buffer(buf_size)
    try:
        glGetActiveAttrib(program_id, index, buf_size, None, asize, atype, aname)
        return (aname.value.decode(), atype.value, asize.value)
    except GLException as exc:
        raise ShaderException from exc