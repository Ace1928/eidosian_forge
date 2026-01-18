import json
import struct
import pyglet
from pyglet.gl import GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_FLOAT
from pyglet.gl import GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER, GL_ARRAY_BUFFER, GL_TRIANGLES
from .. import Model, Material, MaterialGroup
from . import ModelDecodeException, ModelDecoder
class BufferView:

    def __init__(self, buffer, offset, length, target, stride):
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.target = target
        self.stride = stride