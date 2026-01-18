import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _introspect_attributes(program_id: int) -> dict:
    """Introspect a Program's Attributes, and return a dict of accessors."""
    attributes = {}
    for index in range(_get_number(program_id, GL_ACTIVE_ATTRIBUTES)):
        a_name, a_type, a_size = _query_attribute(program_id, index)
        loc = glGetAttribLocation(program_id, create_string_buffer(a_name.encode('utf-8')))
        count, fmt = _attribute_types[a_type]
        attributes[a_name] = dict(type=a_type, size=a_size, location=loc, count=count, format=fmt)
    if _debug_gl_shaders:
        for attribute in attributes.values():
            print(f' Found attribute: {attribute}')
    return attributes