import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _get_uniform_block_name(program_id: int, index: int) -> str:
    """Query the name of a Uniform Block, by index"""
    buf_size = 128
    size = c_int(0)
    name_buf = create_string_buffer(buf_size)
    try:
        glGetActiveUniformBlockName(program_id, index, buf_size, size, name_buf)
        return name_buf.value.decode()
    except GLException:
        raise ShaderException(f'Unable to query UniformBlock name at index: {index}')