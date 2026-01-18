import json
import struct
import pyglet
from pyglet.gl import GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_FLOAT
from pyglet.gl import GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER, GL_ARRAY_BUFFER, GL_TRIANGLES
from .. import Model, Material, MaterialGroup
from . import ModelDecodeException, ModelDecoder
class GLTFModelDecoder(ModelDecoder):

    def get_file_extensions(self):
        return ['.gltf']

    def decode(self, file, filename, batch):
        if not batch:
            batch = pyglet.graphics.Batch()
        vertex_lists = parse_gltf_file(file=file, filename=filename, batch=batch)
        textures = {}
        return Model(vertex_lists, textures, batch=batch)