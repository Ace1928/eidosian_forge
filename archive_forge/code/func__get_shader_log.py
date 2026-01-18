import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _get_shader_log(self, shader_id):
    log_length = c_int(0)
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, byref(log_length))
    result_str = create_string_buffer(log_length.value)
    glGetShaderInfoLog(shader_id, log_length, None, result_str)
    if result_str.value:
        return f"OpenGL returned the following message when compiling the '{self.type}' shader: \n{result_str.value.decode('utf8')}"
    else:
        return f"{self.type.capitalize()} Shader '{shader_id}' compiled successfully."